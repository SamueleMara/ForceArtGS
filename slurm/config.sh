# ForceArtGS on SLURM: the ONE file to edit on the server. Sourced by submit_all.sh and every job.
# ---------------------------------------------------------------- paths (server)
export FORCEARTGS_REPO="${FORCEARTGS_REPO:-$HOME/ForceArtGS}"                         # this checkout
export FORCEARTGS_PYTHON="${FORCEARTGS_PYTHON:-$HOME/miniconda3/envs/forceartgs/bin/python}"   # the forceartgs env
export FORCEARTGS_REAL_ROOT="${FORCEARTGS_REAL_ROOT:-/path/to/Sapien/output/real}"    # the real dataset (episode dirs)
export FORCEARTGS_OUTPUT_ROOT="${FORCEARTGS_OUTPUT_ROOT:-/path/to/outputs/}"          # every output must be under it
export FORCEARTGS_OUT="${FORCEARTGS_OUT:-${FORCEARTGS_OUTPUT_ROOT%/}/forceartgs_all_episodes}"  # this sweep's root
export FORCEARTGS_CONTACT_DIR="${FORCEARTGS_CONTACT_DIR:-$FORCEARTGS_OUT/contact_intervals}"   # never the repo copy
# ---------------------------------------------------------------- what to run
BUDGET="${BUDGET:-full}"                  # smoke | coarse | full
EPISODES="${EPISODES:-all}"               # all = every *_episode_XXXXXX ; first = only 000000
OBJECTS_REGEX="${OBJECTS_REGEX:-.}"       # e.g. '^final_' to run only the final_* objects
PINN_DEVICE="${PINN_DEVICE:-cpu}"         # step-3 PINNs (cpu measured faster for these small nets)
# ---------------------------------------------------------------- SLURM resources (per episode job: 1 GPU)
SLURM_PARTITION="${SLURM_PARTITION:-}"    # empty = cluster default
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
SLURM_QOS="${SLURM_QOS:-}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"      # one episode, full budget (laptop 8 GB GPU: ~3-4 h; requeue resumes)
SLURM_CPUS="${SLURM_CPUS:-8}"
SLURM_MEM="${SLURM_MEM:-64G}"
SLURM_GRES="${SLURM_GRES:-gpu:1}"
SLURM_MAX_PARALLEL="${SLURM_MAX_PARALLEL:-8}"   # episodes running at once (= GPUs you may use)
SLURM_EXTRA="${SLURM_EXTRA:-}"            # anything else for sbatch, e.g. "--constraint=a100"
# ---------------------------------------------------------------- cross-demo validation (after all episodes)
CROSS_DEMO="${CROSS_DEMO:-1}"             # 1 = submit the N x N transfer + report job after the episode array
CROSS_DEMO_MANIFEST="${CROSS_DEMO_MANIFEST:-$FORCEARTGS_REPO/configs/g9_real/object_group_manifest.yaml}"
SYNTHETIC="${SYNTHETIC:-1}"               # 1 = submit the synthetic known-M,c benchmark (independent, starts at once)
CROSS_DEMO_TIME="${CROSS_DEMO_TIME:-24:00:00}"
