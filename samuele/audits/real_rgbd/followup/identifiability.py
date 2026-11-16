"""E4: rank and held-out diagnostic arms; never production calibration."""
from collections import defaultdict
import numpy as np
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .common import atomic_json,csv_write,intervals,population,stable

PARAMETERS=['tx_m','ty_m','tz_m','rx_rad','ry_rad','rz_rad','depth_offset_m','depth_scale']
SCALES=np.array([.01,.01,.01,.005,.005,.005,.01,.01])


def residual_and_jacobian(local_points,rays,pose,reference_normal,reference_offset):
    """T' rotates about its optical center in world axes; t'=t+delta_t.

    Only the target depth varies: D'=(1+s)D+b; reference camera is the gauge.
    """
    R=pose[:3,:3];t=pose[:3,3];P=local_points@R.T;n=np.asarray(reference_normal)
    residual=(P+t)@n-reference_offset
    J=np.c_[np.repeat(n[None],len(P),axis=0),np.cross(P,n),rays@R.T@n,P@n]
    return residual,J


def perturbed_residual(local_points,rays,pose,n,d,delta):
    corrected=local_points*(1+delta[7])+rays*delta[6]
    world=corrected@pose[:3,:3].T@Rotation.from_rotvec(delta[3:6]).as_matrix().T+pose[:3,3]+delta[:3]
    return world@n-d


def rank_diagnostic(J):
    J=np.asarray(J,float);scaled=J*SCALES
    if not len(J):return dict(status='POSE_VS_DEPTH_NOT_IDENTIFIABLE',rank=0,parameters=PARAMETERS,singular_values=[])
    singular=np.linalg.svd(scaled,compute_uv=False);threshold=max(singular[0]*1e-6,1e-10);rank=int((singular>threshold).sum())
    norms=np.linalg.norm(scaled,axis=0);correlation=(scaled.T@scaled)/np.maximum(norms[:,None]*norms[None,:],1e-30)
    gram_values,gram_vectors=np.linalg.eigh(scaled.T@scaled)
    null=(gram_vectors[:,gram_values<threshold**2]*SCALES[:,None]).T
    return dict(status='CONDITIONAL_FULL_RANK' if rank==8 else 'POSE_VS_DEPTH_NOT_IDENTIFIABLE',rank=rank,parameters=PARAMETERS,
        singular_values=singular.tolist(),condition_number=float(singular[0]/singular[-1]) if len(singular)==8 and singular[-1]>1e-15 else None,
        parameter_column_cosines=correlation.tolist(),parameter_scales=SCALES.tolist(),reference_camera_fixed=True,
        approximate_null_parameter_combinations=null.tolist(),
        caveat='rank is necessary, not independent physical identity or absolute accuracy proof')


def compare_models(observations):
    """One fixed population; split physical track AND family, never pixel folds."""
    if not observations:return dict(status='UNIDENTIFIABLE',reason='NO_AUTOMATIC_MULTICUE_SURFACE_SUPPORT',rank=rank_diagnostic(np.empty((0,8)))),[]
    J=np.array([o['jacobian'] for o in observations]);r=np.array([o['residual_m'] for o in observations]);families=[o['family'] for o in observations]
    surfaces=[o['surface_key'] for o in observations]
    hold=np.array([int(stable(f)[:8],16)%3==0 for f in families])
    # If families have no independent holdout, do not manufacture a pixel split.
    train=~hold;rank=rank_diagnostic(J[train] if train.any() else J)
    groups=defaultdict(list)
    for i,o in enumerate(observations):groups[o['surface_key']].append(i)
    rank.update(numerical_rank_status=rank['status'],automatic_track_clusters=len(groups),object_families=len(set(families)))
    if len(groups)<3 or len(set(families))<2:
        rank.update(status='POSE_VS_DEPTH_NOT_IDENTIFIABLE',reason='INSUFFICIENT_INDEPENDENT_SURFACE_AND_FAMILY_SUPPORT; pixel-level fitting noise may lift numerical rank')
    weights=np.array([1/len(groups[k]) for k in surfaces]);models=[]
    for name,cols in [('M0',[]),('Mpose',list(range(6))),('Mbias',[6]),('Mscale',[7]),('Mpose+bias',list(range(7)))]:
        output=dict(model=name,train_points=int(train.sum()),heldout_points=int(hold.sum()),independent_surfaces=len(groups),
            train_families=sorted(set(np.array(families)[train])),heldout_families=sorted(set(np.array(families)[hold])),
            physical_identity='AUTOMATIC_CONDITIONAL',production_parameters_changed=False)
        if not train.any() or not hold.any():models.append(dict(output,status='UNIDENTIFIABLE_NO_DISJOINT_FAMILY_HOLDOUT',coefficients=None));continue
        beta=np.zeros(8)
        if cols:
            design=J[train][:,cols];A=design*np.sqrt(weights[train,None]);b=-r[train]*np.sqrt(weights[train])
            if np.linalg.matrix_rank(A*SCALES[cols],tol=1e-8)<len(cols):models.append(dict(output,status='UNIDENTIFIABLE_RANK_DEFICIENT',coefficients=None));continue
            limits=np.array([.01,.01,.01,np.deg2rad(.3),np.deg2rad(.3),np.deg2rad(.3),.02,.03])[cols]
            fit=lsq_linear(A,b,bounds=(-limits,limits));beta[cols]=fit.x
            if np.any(np.abs(fit.x)>=.999*limits):models.append(dict(output,status='BOUND_HIT',coefficients=beta.tolist()));continue
        after=r+J@beta;effects=[];clusters=[]
        for key,index in groups.items():
            index=np.array(index);index=index[hold[index]]
            if len(index):effects.append(float(np.mean(np.abs(r[index]))-np.mean(np.abs(after[index]))));clusters.append(key)
        output.update(status='CONDITIONAL_DIAGNOSTIC_NOT_CALIBRATION',coefficients=beta.tolist(),parameter_names=PARAMETERS,
            heldout_baseline=population(r[hold]),heldout_after=population(after[hold]),heldout_improvement=intervals(effects,clusters),
            parameter_count=len(cols),train_weighted_squared_error=float(np.sum(weights[train]*after[train]**2)),
            complexity_penalty_aic=float(train.sum()*np.log(max(np.mean(after[train]**2),1e-20))+2*len(cols)),
            uncertainty='bootstrap physical automatic tracks; family split; identity and mounting covariance remain unverified')
        models.append(output)
    return rank,models


def write_target_sop(out):
    text='''# Optional independent metric target experiment\n\nREQUIRES_CAPTURE for absolute accuracy; existing-recording analyses continue.\n\nUse a rigid matte planar board with measured square/tag coordinates in meters. Record actual measured dimensions, measurement uncertainty, instrument and provenance independently of printed nominal dimensions. Keep cameras fixed. Acquire shared RGB-D observations at near/far distances, at least three normal orientations and image-center/corner positions. Preserve per-stream hardware stamps and clock-domain metadata if available.\n\nThe offline target solver accepts 3D measured board corners and their RGB image coordinates. It fixes K and distortion, solves board pose from RGB only, checks reprojection and coordinate provenance, predicts metric camera-Z from ray/plane intersection, and compares registered and native depth separately using T_color_depth. It never fits the reference plane to the depth. A disagreement between declared nominal and measured dimensions invalidates the reference. Project board pose into world coordinates independently per camera; compare normal/offset and pose closure across views.\n'''
    p=out/'calibration/target_capture_required.md';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);return p


def solve_metric_target(object_points,image_points,K,dimension_provenance):
    import cv2
    if not dimension_provenance.get('measured_by') or dimension_provenance.get('units')!='m' or not dimension_provenance.get('independently_measured'):
        return dict(status='REFERENCE_PROVENANCE_UNVERIFIED')
    nominal=dimension_provenance.get('nominal_square_m');measured=dimension_provenance.get('measured_square_m')
    if measured is None or measured<=0 or nominal is not None and abs(nominal-measured)>dimension_provenance.get('uncertainty_m',.0001):return dict(status='REFERENCE_DIMENSION_CONTRADICTED')
    xyz=np.asarray(object_points,float);uv=np.asarray(image_points,float)
    ok,rot,t=cv2.solvePnP(xyz,uv,np.asarray(K,float),np.zeros(5),flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:return dict(status='UNOBSERVABLE_PNP_FAILED')
    reproj=cv2.projectPoints(xyz,rot,t,K,np.zeros(5))[0].reshape(-1,2);R=cv2.Rodrigues(rot)[0];n=R[:,2];d=float(n@t.ravel())
    return dict(status='CONDITIONAL_RGB_METRIC_REFERENCE',rotation=R.tolist(),translation_m=t.ravel().tolist(),normal_camera=n.tolist(),plane_offset_m=d,
        reprojection_rms_px=float(np.sqrt(np.mean(np.sum((reproj-uv)**2,axis=1)))),depth_used_for_reference=False)
