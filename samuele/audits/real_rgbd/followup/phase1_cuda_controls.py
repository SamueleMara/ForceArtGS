#!/usr/bin/env python3
"""Positive/negative synthetic controls using the actual unchanged CUDA model."""
import argparse
import dataclasses
from pathlib import Path
import sys

import numpy as np
from phase1_worker import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();sys.path.insert(0,str(a.code.resolve()))
    import torch
    if not torch.cuda.is_available():write_json(a.out,dict(status='NOT_RUN',reason='CUDA_UNAVAILABLE'));return 2
    from tests.wrong_camera_fixtures import masked_scene
    from estimator.renderer_bridge import build_linearization
    from estimator.sparse_iekf import IEKFConfig,_frame_terms,measurement_operators,iekf_frame_step
    from estimator import wrong_camera_audit as wa
    cams,st,truth,obs,gt,masks,model,_=masked_scene(n=32,shift=(0,0,0))
    cfg=IEKFConfig(max_iters=4,pcg_max_iters=100,camera_names=('cam_000','cam_001','cam_002'),use_rgb=False)
    lin=build_linearization(st,cams,eps_depth=cfg.eps_depth)
    terms=_frame_terms(lin.views,gt,cfg)
    rhs,normal,_=measurement_operators(lin,[t.as_prec4() for t in terms])
    results=[]
    zero=float(rhs.abs().max());results.append(dict(test='exact_static_depth_zero_force',status='PASS' if zero<1e-6 else 'FAIL',max_abs_rhs=zero))
    P=torch.eye(12,device='cuda')[None].repeat(st.num_gaussians,1,1)*1e-3
    state,post,diag=iekf_frame_step(st,P,gt,cams,dt=.04,cfg=cfg)
    drift=float((state.mu-st.mu).abs().max());results.append(dict(test='static_perfect_depth_no_false_motion',status='PASS' if drift<1e-6 else 'FAIL',max_abs_displacement_m=drift))
    biased=list(gt);rgb,D,V=biased[0];biased[0]=(rgb,D+.005,V)
    terms_b=_frame_terms(lin.views,biased,cfg);rb,_,_=measurement_operators(lin,[t.as_prec4() for t in terms_b]);mag=float(rb.norm())
    results.append(dict(test='known_positive_5mm_bias_has_actual_depth_force',status='PASS' if mag>1e-6 else 'FAIL',rhs_norm=mag))
    no_depth=[(rgb,D,torch.zeros_like(V,dtype=torch.bool)) for rgb,D,V in gt]
    empty=_frame_terms(lin.views,no_depth,cfg);r0,_,_=measurement_operators(lin,[t.as_prec4() for t in empty]);z=float(r0.abs().max())
    results.append(dict(test='no_core_no_valid_depth_is_zero_information',status='PASS' if z==0 else 'FAIL',max_abs_rhs=z))
    cfg_rgb=dataclasses.replace(cfg,use_rgb=True);wrong=list(gt);rgb,D,V=wrong[1];wrong[1]=(torch.flip(rgb,[-1]),D+.15,V)
    tw=_frame_terms(lin.views,wrong,cfg_rgb);bc=wa.camera_channel_rhs(lin,tw)
    results.append(dict(test='wrong_RGB_depth_correspondence_has_separate_channels',status='PASS' if float(bc[1,'rgb'].norm())>0 and float(bc[1,'depth'].norm())>0 else 'FAIL',
        rgb_force_norm=float(bc[1,'rgb'].norm()),depth_force_norm=float(bc[1,'depth'].norm()),synthetic_truth='mismatched RGB and foreground depth layer'))
    # Reuse existing CUDA controls for co-visibility coupling, exact contributor
    # T*alpha, and observer-on/off production-state parity.
    for name in ['test_wrong_camera_force_additivity','test_wrong_camera_grouped_loo','test_wrong_camera_contributor_parity']:
        module=__import__('tests.'+name,fromlist=['main']);rc=module.main();results.append(dict(test=name,status='PASS' if rc==0 else 'FAIL',returncode=rc))
    passed=all(r['status']=='PASS' for r in results);write_json(a.out,dict(status='PASS' if passed else 'FAIL',results=results,actual_renderer=True,production_modified=False))
    return 0 if passed else 1


if __name__=='__main__':sys.exit(main())
