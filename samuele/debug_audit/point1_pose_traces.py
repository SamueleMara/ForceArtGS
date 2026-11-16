#!/usr/bin/env python3
"""Point 1 -- physical vs virtual cameras in a prepared multiview dataset (read only).

transforms_0.json carries the PHYSICAL calibration per camera (must be constant over time, as must K);
transforms_1.json carries VIRTUAL cameras T_part_camera_virtual (moving-part canonical frame), which may vary.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

mv, out = Path(sys.argv[1]), Path(sys.argv[2])
res = {}
for part, label in ((0, "T_physical_camera (static part)"), (1, "T_part_camera_virtual (moving part)")):
    tj = json.loads((mv / f"transforms_{part}.json").read_text())
    by = defaultdict(list)
    for fr in tj["frames"]:
        by[fr["camera_name"]].append(fr)
    cams = {}
    for c, frs in by.items():
        T = np.array([f["transform_matrix"] for f in frs])
        K = np.array([[f["fl_x"], f["fl_y"], f["cx"], f["cy"]] for f in frs])
        dR = [float(np.degrees(np.arccos(np.clip((np.trace(T[0][:3, :3].T @ t[:3, :3]) - 1) / 2, -1, 1)))) for t in T]
        dt = [float(np.linalg.norm(t[:3, 3] - T[0][:3, 3]) * 1e3) for t in T]
        cams[c] = dict(frames=len(frs), max_abs_matrix_change=float(np.abs(T - T[0]).max()), max_rotation_change_deg=max(dR), max_translation_change_mm=max(dt),
                       max_K_change=float(np.abs(K - K[0]).max()), K=K[0].tolist(),
                       per_frame_rotation_deg=dR, per_frame_translation_mm=dt)
    res[f"part_{part}"] = dict(meaning=label, cameras=cams)
res["shared_cam_K_txt"] = np.loadtxt(mv / "cam_K.txt").tolist()
res["multiview_json"] = {k: v for k, v in json.loads((mv / "multiview.json").read_text()).items() if k != "frames"}
res["checks"] = dict(physical_cameras_fixed=all(v["max_abs_matrix_change"] == 0 and v["max_K_change"] == 0
                                                for v in res["part_0"]["cameras"].values()),
                     virtual_cameras_vary=any(v["max_rotation_change_deg"] > 0 for v in res["part_1"]["cameras"].values()),
                     schema_marks_virtual_pose="NO: transforms_1.json uses the same 'transform_matrix' key and schema as the physical part 0")
out.mkdir(parents=True, exist_ok=True)
(out / "physical_vs_virtual_pose_traces.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res["checks"]), {c: (round(v["max_rotation_change_deg"], 2), round(v["max_translation_change_mm"], 1)) for c, v in res["part_1"]["cameras"].items()})
