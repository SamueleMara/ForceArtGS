# ForceArtGS on a SLURM cluster: every episode of every object, then the cross-demo validation

One array task per episode (1 GPU each) runs the whole chain: reconstruction → q(t) extension → M, c estimation →
URDF/USD with the 3DGS parts. The summary job runs after the array. The cross-demo validation runs after the summary
(an N × N frozen-parameter PhysX transfer for each object). The synthetic known-M,c benchmark runs independently.
Everything resumes: resubmitting skips finished steps and pairs, and 3DGS training restarts from its checkpoints.

## 1. Set up the server (once)

1. **Code:** clone this repository, branch `multimodal-tcp-contact`, with its nerfstudio submodule (`reconstruction/nerfstudio`).
   The submodule fix in `nerfstudio/utils/eval_utils.py` (`config.load_step = None`) must be present. It makes the export
   after a requeued or resumed training use the latest checkpoint.
2. **Env:** `envs/create_forceartgs_env.sh` builds the self-contained `forceartgs` env. On the server:
   - Set `TORCH_CUDA_ARCH_LIST` to your GPUs (for example `8.0` for A100, `9.0` for H100). The default `8.9` is the laptop's.
   - The script clones the CUDA stack from a `freeartgs` env (`FREEARTGS_ENV`). That env must exist on the server,
     or you install the same pinned stack (`envs/forceartgs.pip-freeze.txt`).
   - ForceArtGS-Physics is installed from `FORCEARTGS_PHYSICS_SRC`, which is your private copy (no redistribution).
3. **Weights:** run `bash scripts/download_checkpoints.sh`. This puts the weights in `checkpoints/`, sha256-checked.
4. **Data:** copy the real dataset (the folder of `*_episode_XXXXXX` directories) to the server.
5. **Check:** run `scripts/run_tests.sh tests/test_g9_cross_demo_pipeline.py tests/test_g9_cross_demo_metrics.py`.

## 2. Configure

Edit `slurm/config.sh`. It is the only file that holds server paths:

- `FORCEARTGS_REPO`, `FORCEARTGS_PYTHON`
- `FORCEARTGS_REAL_ROOT`
- `FORCEARTGS_OUTPUT_ROOT`: every output must be under it.
- `FORCEARTGS_OUT`: this sweep.

It also holds `BUDGET`, `EPISODES`, `OBJECTS_REGEX` and the SLURM resources: partition, account, time, CPUs, memory,
`SLURM_MAX_PARALLEL`.

Contact segments are written to `$FORCEARTGS_OUT/contact_intervals` (`FORCEARTGS_CONTACT_DIR`), never to the repo.

**Object identity, which you must confirm:** `configs/g9_real/object_group_manifest.yaml` lists the candidate groups
found on disk, all with `verified_same_physical_object: null`. Set `true` for each group that is the same physical
object with the same fixed base and sensor setup. The cross-demo job refuses unverified groups and reports
`OBJECT_IDENTITY_UNVERIFIED`.

## 3. Submit and monitor

```bash
DRY_RUN=1 bash slurm/submit_all.sh     # prints the episode list and the sbatch commands
bash slurm/submit_all.sh               # submits the array, the summary, the cross-demo job and the synthetic job
squeue -u $USER
tail -f "$FORCEARTGS_OUT/real_full_to_estimated_assets.log"          # one line per scene step
ls "$FORCEARTGS_OUT/slurm_logs/"                                     # ep_<job>_<task>.out per episode
cat "$FORCEARTGS_OUT/estimated_assets/SUMMARY.md"                    # rebuilt after every episode
cat "$FORCEARTGS_OUT/cross_demo_validation/v*/report.md"             # after the cross-demo job
```

Cancel everything with `scancel -n forceartgs_ep -n forceartgs_summary -n forceartgs_cross_demo -n forceartgs_synthetic`.
After a time limit, a failure or a cancel, run the same `bash slurm/submit_all.sh` again to resume.
