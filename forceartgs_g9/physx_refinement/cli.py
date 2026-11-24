"""ForceArtGS estimator-side, isolated diagnostic G10 PhysX refinement CLI.

python -m forceartgs_g9.physx_refinement.cli --urdf object.urdf \
  --episodes-json g9_physx_episodes.json --config-json g9_physx_config.json \
  --out g9/physx_refinement.json

Input episodes originate from explicit G9Interaction regime annotations;
no PArtTwin1D dependency, no simulator asset writer, no promotion of priors.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .objective import Episode
from .sapien_diagnostic import IsolatedSapienCpu

def run(urdf: Path, episodes_json: Path, config_json: Path, out: Path) -> dict:
    d=json.loads(episodes_json.read_text())
    c=json.loads(config_json.read_text())
    episodes=[Episode(name=e["name"],t_s=np.asarray(e["t_s"]),q=np.asarray(e["q"]),
                      sigma_q=np.asarray(e["sigma_q"]),q0=float(e["q0"]),v0=float(e["v0"]),
                      input_t_s=np.asarray(e["input_t_s"]),input_effort=np.asarray(e["input_effort"]),
                      mode=e["mode"],calibration_status=e.get("calibration_status","UNVERIFIED"),
                      used_for_training=bool(e.get("used_for_training",True)),
                      exploratory_assumptions_accepted=bool(e.get("exploratory_assumptions_accepted",False)),
                      sigma_v0=e.get("sigma_v0")) for e in d["episodes"]]
    if not d.get("regime_provenance"):
        raise ValueError("episode JSON requires explicit regime_provenance")
    ph=c["physics"]
    if ph.get("engine") != "sapien_cpu_diagnostic":
        raise ValueError("only sapien_cpu_diagnostic implemented; native Isaac PhysX runner NOT available")
    from .api import estimate
    runner=IsolatedSapienCpu(urdf,moving_link=ph["moving_link"],joint_name=ph["joint_name"],
                            dt=float(ph["timestep"]),gravity=ph.get("gravity",[0,0,-9.81]),
                            friction_mode=ph["friction_mode"],python=ph.get("python"))
    try:
        fit=estimate(runner,episodes,physics=ph,loss=c.get("loss",{}),fitting=c["fitting"],
                     profile=c.get("profile"),prior_rho_range=c.get("prior_rho_range"),log=print)
    finally:
        runner.close()
    fit["joint_native_properties_suggested_only"]=fit["native_properties_suggestions_only"]
    fit["provenance"]={"project":"ForceArtGS", "urdf":str(urdf.resolve()),
        "urdf_sha256":hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "episodes_sha256":hashlib.sha256(episodes_json.read_bytes()).hexdigest(),
        "config_sha256":hashlib.sha256(config_json.read_bytes()).hexdigest(),
        "backend":"SAPIEN_CPU_DIAGNOSTIC", "friction_mode":ph["friction_mode"]}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(fit,indent=2,default=str))
    return fit

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--urdf",required=True,type=Path)
    ap.add_argument("--episodes-json",required=True,type=Path)
    ap.add_argument("--config-json",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    r=run(a.urdf,a.episodes_json,a.config_json,a.out)
    print(json.dumps({"out":str(a.out),"status":r["status"],"loss":r["loss"]["objective"]},indent=2))

if __name__=="__main__":
    main()
