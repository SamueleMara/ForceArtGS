"""Scene x hypothesis x variant x estimator comparison (CSV / JSON / MD) from the latest forcetwin_adaptation
version of every run.   python tools/g9_forcetwin_comparison.py [RUN_ROOT] [OUT_DIR]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from forceartgs_g9.cli import main  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "/media/samuelemara/Extreme SSD/Samuele/results_19_11_26/forceartgs_joint_aware"
out = sys.argv[2] if len(sys.argv) > 2 else "/media/samuelemara/Extreme SSD/Samuele/results_21_11_26/forcetwin_comparison"
sys.exit(main(["compare-dynamics", "--root", root, "--out", out]))
