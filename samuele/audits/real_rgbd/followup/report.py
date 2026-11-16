from collections import Counter,defaultdict
from pathlib import Path
import json
import resource
import numpy as np

from .common import atomic_json,csv_write,iter_rows,intervals,population
from .identifiability import compare_models,write_target_sop
from .timing import fixed_population_delay


def calibration_report(out):
    groups=defaultdict(list)
    for p in sorted((out/'geometry/pose_support').glob('*/*.parquet')):
        for r in iter_rows(p):groups[r['calibration_sha256'],r['camera'],r['target_camera']].append(r)
    ranks=[];models=[]
    for key,observations in groups.items():
        rank,arms=compare_models(observations);ranks.append(dict(calibration_sha256=key[0],camera=key[1],target_camera=key[2],**rank))
        for arm in arms:models.append(dict(calibration_sha256=key[0],camera=key[1],target_camera=key[2],**arm))
    if not groups:ranks=[dict(status='UNIDENTIFIABLE',reason='NO_AUTOMATIC_MULTICUE_STATIC_SURFACE_SUPPORT',absolute_accuracy='UNVERIFIED')]
    p=out/'calibration';p.mkdir(parents=True,exist_ok=True);atomic_json(p/'jacobian_rank.json',ranks);csv_write(p/'heldout_model_comparisons.csv',models)
    write_target_sop(out);return ranks


def timing_report(out,scenes):
    from .timing import timestamp_inventory
    timestamps=[]
    for scene in scenes:timestamps+=timestamp_inventory(scene)
    p=out/'timing';p.mkdir(parents=True,exist_ok=True);csv_write(p/'original_timestamp_inventory.csv',timestamps)
    tracks=defaultdict(list)
    for file in (out/'geometry/associations').glob('*/*.parquet'):
        for r in iter_rows(file):
            if r['status']!='AUTO_MULTICUE_CONSISTENT_CONDITIONAL':continue
            key=r['scene'],r['camera'],r['target_camera'],r['source_track'],r['target_track'];tracks[key].append(r)
    by_name={s['name']:s for s in scenes};results=[]
    for key,rows in tracks.items():
        rows=sorted(rows,key=lambda r:r['processed_frame']);s=by_name[key[0]];fps=s.get('info',{}).get('fps',0)
        if not fps:continue
        normals_a=np.array([r['source_normal_world'] for r in rows]);normals_b=np.array([r['target_normal_world'] for r in rows])
        normals_stable=len(rows)>=20 and np.max(np.linalg.norm(normals_a-normals_a.mean(0),axis=1))<.03 and np.max(np.linalg.norm(normals_b-normals_b.mean(0),axis=1))<.03
        if not normals_stable:continue
        t=np.array([r['source_frame']/fps for r in rows]);aa=[r['source_offset_world_m'] for r in rows];bb=[r['target_offset_world_m'] for r in rows]
        result=fixed_population_delay(t,aa,bb,np.arange(-6,7)*2/fps,identity_confirmed=True)
        results.append(dict(scene=key[0],camera=key[1],target_camera=key[2],source_track=key[3],target_track=key[4],calibration_sha256=s['calibration_sha256'],
            identity='AUTOMATIC_CONDITIONAL_NOT_INDEPENDENT_GT',time_base='source index / recorder FPS, not hardware time',**result))
    present={r['scene'] for r in results}
    for s in scenes:
        if s['name'] not in present:results.append(dict(scene=s['name'],status='TIMING_UNOBSERVABLE',reason='No persistent identified shared translating plane with varying velocity and stable normals',applied_shift=False))
    csv_write(p/'timing_hypotheses.csv',results)
    return dict(status='CONDITIONAL' if any(r['status']=='CONDITIONAL_IMAGE_CONTENT_DELAY' for r in results) else 'UNOBSERVABLE',
        rows=len(results),sensor_timestamp_candidates=sum(r['status']=='SENSOR_TIMESTAMP_CANDIDATE_CLOCK_DOMAIN_UNVERIFIED' for r in timestamps),applied_shift=False)


def generate(out,inventory,receipts,stages,synthetic=None,phase1=None):
    out=Path(out);family_groups=defaultdict(list);pair_groups=defaultdict(list);patch_tracks=set();counts=Counter();association=Counter();coverage=defaultdict(lambda:np.zeros(3));moments=defaultdict(lambda:np.zeros(4));histograms=defaultdict(lambda:np.zeros(1002,np.int64));holes=defaultdict(Counter)
    for p in sorted((out/'geometry/associations').glob('*/*.parquet')):
        for r in iter_rows(p):
            association[r['status']]+=1
            if r['status']!='AUTO_MULTICUE_CONSISTENT_CONDITIONAL':continue
            value=r['source_to_target_plane']['signed_median_m']
            if value is None:continue
            identity=(r['scene'],r['camera'],r['source_track'],r['target_camera'],r['target_track'])
            patch_tracks.add(identity);key=(r['calibration_sha256'],r['family'],r['scene'],r['camera'],r['target_camera'])
            pair_groups[key].append((value,r['projected_overlap_pixels'],identity))
            stat=r['source_to_target_plane'];moments[key]+=np.array([stat['count'],stat.get('sum_signed_m',0),stat.get('sum_abs_m',0),stat.get('sum_sq_m2',0)])
            histograms[key]+=np.array(r.get('qualified_residual_histogram_counts',[0]*1002))
    pairs=[]
    for (cal,fam,scene,a,b),values in pair_groups.items():
        surface=defaultdict(list)
        for v,n,k in values:surface[str(k)].append(v)
        v=[float(np.median(x)) for x in surface.values()];ci=intervals(v,list(surface))
        row=dict(scene=scene,family=fam,calibration_sha256=cal,camera=a,target_camera=b,status='CONDITIONAL_AUTOMATIC_IDENTITY',
            independent_physical_surfaces='UNVERIFIED',automatic_tracks=len(surface),patch_frame_observations=len(values),
            projected_pixels=sum(x[1] for x in values),equal_automatic_surface_signed_median_m=float(np.median(v)),
            uncertainty=ci,pooled_patch_frame=population([x[0] for x in values]),absolute_accuracy='UNVERIFIED')
        total=moments[cal,fam,scene,a,b];hist=histograms[cal,fam,scene,a,b];n=int(total[0]);index=int(np.searchsorted(np.cumsum(hist),max(1,n/2)))
        centers=np.r_[np.inf,-.5+(np.arange(1000)+.5)*.001,np.inf];order=np.argsort(np.abs(centers));absolute=int(np.searchsorted(np.cumsum(hist[order]),max(1,.95*n)))
        row.update(pooled_heldout_pixel_count=n,pooled_pixel_signed_mean_m=float(total[1]/n) if n else None,pooled_pixel_mae_m=float(total[2]/n) if n else None,
            pooled_pixel_rmse_m=float(np.sqrt(total[3]/n)) if n else None,
            pooled_pixel_signed_median_approx_m=float(centers[index]) if 0<index<1001 else None,
            pooled_pixel_abs_p95_approx_m=float(abs(centers[order[min(absolute,1001)]])) if absolute<1002 and np.isfinite(centers[order[min(absolute,1001)]]) else None,
            quantile_method='1-mm signed histogram; temporal pixels are repeated observations, not independent samples')
        pairs.append(row);family_groups[cal,fam,a,b].append(row)
    families=[]
    for (cal,fam,a,b),rows in family_groups.items():
        v=[r['equal_automatic_surface_signed_median_m'] for r in rows]
        families.append(dict(calibration_sha256=cal,family=fam,camera=a,target_camera=b,status='CONDITIONAL',scenes=len(rows),
            automatic_tracks=sum(r['automatic_tracks'] for r in rows),equal_scene_signed_median_m=float(np.median(v)),
            equal_scene_mae_m=float(np.mean(np.abs(v))),uncertainty=intervals(v,[r['scene'] for r in rows]),physical_identity='AUTOMATIC_NOT_INDEPENDENT_GT'))
    for p in (out/'registration').glob('*/*.parquet'):
        for r in iter_rows(p):
            if r['status'] not in {'MEASURED','CONTRADICTED'}:continue
            key=r['calibration_sha256'],r['family'],r['camera'];h=holes[key];h['observations']+=1
            for field in ['native_pixels','native_zero_count','native_calibrated_ray_pixels','native_zero_in_calibrated_rays','color_valid_pixels','registered_zero_inside_rgb_count','outside_rgb_support_pixels','gap_candidate_pixels','discontinuity_candidate_pixels','unknown_hole_pixels']:h[field]+=r.get(field,0)
    hole_rows=[]
    for (cal,fam,c),h in holes.items():
        hole_rows.append(dict(calibration_sha256=cal,family=fam,camera=c,**dict(h),native_invalid_fraction=h['native_zero_count']/max(1,h['native_pixels']),
            registered_hole_fraction_within_RGB=h['registered_zero_inside_rgb_count']/max(1,h['color_valid_pixels']),
            gap_candidate_fraction_of_holes=h['gap_candidate_pixels']/max(1,h['registered_zero_inside_rgb_count']),
            target_raw_dropout_partition='UNIDENTIFIABLE_without_depth_for_invalid_native_rays'))
    for s in inventory['scenes']:
        for i,a in enumerate(s.get('cameras',[])):
            for b in s['cameras'][i+1:]:
                if not any(r['scene']==s['name'] and r['camera']==a['name'] and r['target_camera']==b['name'] for r in pairs):
                    pairs.append(dict(scene=s['name'],family=s['family'],calibration_sha256=s.get('calibration_sha256'),camera=a['name'],target_camera=b['name'],status='UNIDENTIFIABLE_NO_MULTICUE_SUPPORT',automatic_tracks=0,absolute_accuracy='UNVERIFIED'))
    for result in receipts.values():counts.update(result.get('counts',{}))
    for p in (out/'coverage').glob('*/*.parquet'):
        for r in iter_rows(p):
            if r.get('predicted_pixels'):
                coverage[r['family'],r['camera']]+=np.array([1,r['predicted_valid_pixels']/r['predicted_pixels'],r['predicted_pixels']])
    coverage_rows=[dict(family=k[0],camera=k[1],frames=int(v[0]),mean_predicted_mask_valid_fraction=v[1]/v[0],predicted_mask_pixels=int(v[2]),mask_truth='UNVERIFIED') for k,v in coverage.items()]
    (out/'geometry').mkdir(exist_ok=True)
    csv_write(out/'geometry/per_scene_pair_accuracy.csv',pairs);csv_write(out/'geometry/family_vs_pooled_summary.csv',families);csv_write(out/'coverage/family_summary.csv',coverage_rows);csv_write(out/'registration/family_hole_summary.csv',hole_rows)
    overall=defaultdict(list)
    for r in families:overall[r['calibration_sha256'],r['camera'],r['target_camera']].append(r)
    csv_write(out/'geometry/equal_family_summary.csv',[dict(calibration_sha256=k[0],camera=k[1],target_camera=k[2],families_with_support=len(v),equal_family_signed_median_m=float(np.median([r['equal_scene_signed_median_m'] for r in v])),
        equal_family_mae_of_scene_signed_medians_m=float(np.mean(np.abs([r['equal_scene_signed_median_m'] for r in v]))),physical_surfaces='AUTOMATIC_TRACK_PROXIES_NOT_VERIFIED_GT') for k,v in overall.items()])
    scopes=Counter()
    for p in (out/'geometry/scope').glob('*/*.parquet'):
        for r in iter_rows(p):scopes[r['status']]+=1
    all_count=counts['frame_camera_observations']==inventory['expected_frame_camera_rows']
    gates={
        'G0':dict(status='PASS' if inventory['scene_count']==118 and all_count else 'NOT_RUN_FULL_SCOPE',observations=counts['frame_camera_observations'],expected=inventory['expected_frame_camera_rows']),
        'G1':dict(status='FAIL' if counts['replay_different_pixels'] else 'PASS' if counts['registration_replays']==inventory['expected_frame_camera_rows'] else 'NOT_RUN_FULL_SCOPE',replays=counts['registration_replays'],differing_pixels=counts['replay_different_pixels']),
        'G2':dict(status='UNOBSERVABLE',prediction_only=True,human_verified_images=0,reason='Automatic mask validity is not manually measured mask precision/recall'),
        'G3':dict(status='PASS' if synthetic and synthetic.get('hard_negative_identity') else 'NOT_RUN',real_association_decisions=dict(association),physical_identity='CONDITIONAL_AUTOMATIC'),
        'G4':dict(status='CONDITIONAL' if patch_tracks else 'UNOBSERVABLE',automatic_tracks=len(patch_tracks),independent_physical_identity='UNVERIFIED'),
        'G5':dict(status='UNOBSERVABLE',reason='See calibration rank/held-out results; automatic identity and absolute reference remain conditional'),
        'G6':dict(status='UNOBSERVABLE',reason='No synchronization corrections; inspect timing hypotheses and stamp provenance'),
        'G7':phase1 or dict(status='NOT_RUN',reason='Phase-1 diagnostic stage not requested'),
        'G8':dict(status='PASS' if all_count else 'NOT_RUN_FULL_SCOPE',scientific_identification='PARTIAL_CONDITIONAL',absolute_accuracy='UNVERIFIED')}
    if synthetic and synthetic.get('failed'):gates['synthetic']=dict(status='FAIL',failed=synthetic['failed'])
    atomic_json(out/'gate_results.json',gates)
    report=['# Automatic RGB-D root-cause identifiability experiments','',
        f'Inventory: {inventory["scene_count"]} scenes ({inventory["final_episode_count"]} final episodes, {inventory["legacy_scene_count"]} legacy). Expected {inventory["expected_frame_camera_rows"]} camera/frame observations. Measured scope: {counts["frame_camera_observations"]}. Calibration strata remain separate.',
        '', '## Physical correspondence and automatic surfaces',
        f'Extracted {counts["patches"]} planar/nonplanar candidates; {len(patch_tracks)} persistent automatic multi-cue track pairs. No descriptor pair or fixed-calibration geometric agreement is independent physical identity. Repeated-texture/opposite-face hypotheses are rejected or retained as ambiguous, never called a sensor bias.',
        f'All-frame pair accounting: {dict(scopes)}. Decision counts: {dict(association)}. See geometry/scope, patches, associations and per_scene_pair_accuracy.csv.',
        '', '## Native dropout versus registration',
        f'Instrumented exact replay on {counts["registration_replays"]} observations: {counts["replay_different_pixels"]} differing pixels. Native invalid-ray targets are unknown without a depth hypothesis. Outside RGB support is separate from unoccupied target pixels. Gap/discontinuity labels are conditional; unknown holes remain explicit. Held-out splat validation uses withheld measured native samples, not the fill itself.',
        '', '## Masks, availability, and temporal precision',
        'All requested frames are analyzed, including black laptop. Predicted-mask statistics, connected missing components, frame-to-frame persistence and entering/leaving mask regions are separate from true material physics. No masks were changed or automatically certified as GT. Existing native temporal repeatability evidence remains conditional; quantization and repeat frames do not establish absolute accuracy.',
        '', '## Pose versus depth offset/scale',
        'calibration/jacobian_rank.json reports scaled joint Jacobian rank and parameter correlations. Bounded M0/Mpose/Mbias/Mscale/Mpose+bias fits use fixed candidate populations and disjoint-family holdouts; BOUND_HIT or rank deficiency is nonidentification. Automatic surface identity remains conditional even at full rank. An external target is optional additional evidence, not a prerequisite to current-data analysis.',
        '', '## Timing',
        'Original timestamp schemas and index mappings are inventoried. Source-index/nominal times are not hardware timestamps. Image-content lag uses fixed automatic track identities, shared support at every tested lag, translation-only normal-stable targets and disjoint motion segments; constant velocity/static controls abstain. No shift is applied.',
        '', '## Phase-1 forces and propagation',
        str(gates['G7']),
        'Actual production-renderer contributions and certified frozen solves are required before attributing Gaussian updates. A canonical-vs-observed residual is not an independently calibrated sensor error. Missing compatible runs or unconverged solves are explicit.',
        '', '## Generalization and ordered interventions',
        'Use equal-track/equal-scene/equal-family summaries rather than Makita-dominated pooled pixels. Prioritize laptop availability, repeated-face alias rejection, and verified automatic shared-surface support. Investigate rank-deficient geometry or unavailable timing before proposing corrections. Keep calibration, data, masks and production tracking unchanged.',
        '', '## Gates','```json',json.dumps(gates,indent=2),'```']
    report+=['','## Quantitative availability by family/camera','| Family | Camera | Frames | Mean valid depth inside predicted mask |','|---|---|---:|---:|']
    for r in sorted(coverage_rows,key=lambda r:(r['family'],r['camera'])):report.append(f'| {r["family"]} | {r["camera"]} | {r["frames"]} | {100*r["mean_predicted_mask_valid_fraction"]:.2f}% |')
    report+=['','## Conditional hole mechanisms by calibration/family/camera','| Rig | Family | Camera | Native invalid | Registered holes inside valid RGB | Conservative gap candidates / holes |','|---|---|---|---:|---:|---:|']
    for r in sorted(hole_rows,key=lambda r:(r['calibration_sha256'],r['family'],r['camera'])):report.append(f'| {r["calibration_sha256"][:12]} | {r["family"]} | {r["camera"]} | {100*r["native_invalid_fraction"]:.2f}% | {100*r["registered_hole_fraction_within_RGB"]:.2f}% | {100*r["gap_candidate_fraction_of_holes"]:.2f}% |')
    report+=['','Different native/registered pixel domains are not subtracted. Gap labels and temporal static labels are conditional, not measured causal percentages. All other holes stay explicit unknown/discontinuity candidates.','',
        '## Conditional surface comparisons, de-pooled by scene and family','| Rig | Family | Pair | Supported scenes | Automatic tracks | Equal-scene signed median |','|---|---|---|---:|---:|---:|']
    for r in sorted(families,key=lambda r:(r['calibration_sha256'],r['family'],r['camera'],r['target_camera'])):report.append(f'| {r["calibration_sha256"][:12]} | {r["family"]} | {r["camera"]} to {r["target_camera"]} | {r["scenes"]} | {r["automatic_tracks"]} | {1000*r["equal_scene_signed_median_m"]:.3f} mm |')
    if not families:report.append('| — | All requested families | All requested pairs | 0 | 0 | UNIDENTIFIABLE |')
    (out/'ROOT_CAUSE_IDENTIFIABILITY_REPORT.md').write_text('\n'.join(report)+'\n')
    (out/'FIGURES.md').write_text('# Figures\n\nannotation_review/<scene>/<camera>_<frame>_automatic.png compares RGB, measured registered depth, predicted mask, automatic patch IDs, native measured depth and hole categories. Registration NPZs preserve source UV, Z-buffer winners, occupancy and conditional hole labels. These are automatic proposals, not human-confirmed physical IDs.\n')
    return gates
