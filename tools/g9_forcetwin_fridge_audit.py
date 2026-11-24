"""Red-fridge signed impulse audit (ForceTwin-adaptation spec section 3): record, RGB-D kinematic hypotheses,
pulse 229-232 impulse-momentum sensitivity study and decision. Writes a NEW version under
<run>/dynamics/forcetwin_adaptation/vNNN (never touches older outputs).
    python tools/g9_forcetwin_fridge_audit.py [--rows 229:232] [--root RESULTS_ROOT]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9.cli import main  # noqa: E402

SSD = Path("/media/samuelemara/Extreme SSD/Samuele")
SC = "final_red_fridge_episode_000000"

ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="229:232")
ap.add_argument("--root", default=str(SSD / "results_19_11_26" / "forceartgs_joint_aware"))
a = ap.parse_args()
run = sorted(Path(a.root).glob(f"{SC}_multiview_*_joint_aware_full"))[0]
disc = SSD / "results_17_11_26" / "overnight_real" / "joint" / SC
sys.exit(main(["fridge-audit", "--run", str(run), "--scene", str(SSD / "Backup_09_10_26/Sapien/output/real" / SC),
               "--robot-archive", str(disc / "robot_archive.npz"),
               "--contact-segments", str(Path(__file__).resolve().parents[1] / f"configs/contact_intervals/{SC}.yaml"),
               "--discovery-dir", str(disc / "NEW_cluster_obs_cam001_002"), "--pulse-rows", a.rows]))
