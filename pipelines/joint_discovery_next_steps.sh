#!/usr/bin/env bash
# Staged, reproducible A/B for the joint-discovery "next steps" sprint (one real final_* scene).
# Read-only on the scene; every stage writes a fresh NEXT_* directory and refuses to overwrite.
#
#   bash pipelines/joint_discovery_next_steps.sh <scene_dir> <frames a:b:s> <cam,cam> [out_root]
#
# Env:
#   PYTHON        interpreter (default: freeartgs env)
#   CONFIG        base config (default configs/joint_discovery.real_tcp2cm.yaml)
#   OBS_DIR       existing observation tables obs_<cam>.npz (default: extract into out_root)
#   DIAG_RUN      optional extra fit run for the Stage-A audit, e.g. a cam_000-only run (role diagnostic)
#   REF_JSON      optional tracker-free reference (POST HOC ONLY, never used for fitting)
#   STAGES        subset of "A0 A1 A3 C2 D1" (default all)
#
# Stages: A0 frozen baseline (RGB-D and 2D-only, legacy selection) -> A1 frozen-hypothesis depth audit ->
#         A3 ablations on identical tracks/folds (cluster-robust z + observability policy) -> C2 blocked-time
#         and held-out-camera folds -> D1 full-episode contact/wrench audit.
set -euo pipefail
SCENE="$1"; FRAMES="$2"; CAMS="$3"
NAME=$(basename "$SCENE")
OUT="${4:-outputs/joint_discovery_next/$NAME}"
PY="${PYTHON:-${FORCEARTGS_PYTHON:-$HOME/miniconda3/envs/forceartgs/bin/python}}"
eval "$("$PY" "$(dirname "${BASH_SOURCE[0]}")/../utils/forceartgs_env.py")"   # clean env: forceartgs + repo only
CONFIG="${CONFIG:-configs/joint_discovery.real_tcp2cm.yaml}"
STAGES="${STAGES:-A0 A1 A3 C2 D1}"
CLI=("$PY" -m estimotion_2part.joint_discovery.cli)
IFS=',' read -ra CAM_LIST <<< "$CAMS"
TAG=$(IFS=_; echo "${CAM_LIST[*]}")
OBS="${OBS_DIR:-$OUT/observations_${FRAMES//:/-}}"
mkdir -p "$OUT" "$OBS"
ARCH="$OUT/robot_archive.npz"
[[ -f "$ARCH" ]] || "${CLI[@]}" robot-archive --scene "$SCENE" --output "$ARCH"
OBS_FILES=()
for cam in "${CAM_LIST[@]}"; do
  f="$OBS/obs_$cam.npz"
  [[ -f "$f" ]] || "${CLI[@]}" extract --scene "$SCENE" --camera "$cam" --frames "$FRAMES" --tcp-npz "$ARCH" --output "$f"
  OBS_FILES+=("$f")
done
sha256sum "${OBS_FILES[@]}" > "$OUT/NEXT_observation_sha256_${TAG}.txt"
has() { [[ " $STAGES " == *" $1 "* ]]; }
B2="$OUT/NEXT_A0_B2_rgbd_$TAG"; B3="$OUT/NEXT_A0_B3_2donly_$TAG"
BASE=(--scene "$SCENE" --config "$CONFIG" --cameras "${CAM_LIST[@]}" --observations "${OBS_FILES[@]}")
if has A0; then
  [[ -d "$B2" ]] || "${CLI[@]}" fit "${BASE[@]}" --experiment-id NEXT_A0_B2 --output "$B2"
  [[ -d "$B3" ]] || "${CLI[@]}" fit "${BASE[@]}" --no-depth --fold-from "$B2" --experiment-id NEXT_A0_B3 --output "$B3"
fi
if has A1; then
  RUNS=("B2=$B2" "B3=$B3")
  [[ -n "${DIAG_RUN:-}" ]] && RUNS+=("D=${DIAG_RUN}:diagnostic")
  EXTRA=()
  [[ -n "${REF_JSON:-}" ]] && EXTRA+=(--reference-json "$REF_JSON")
  PAIRS=(B2_R:B2_P B2_R:B3_R STATIC_null:B2_R)
  [[ -n "${DIAG_RUN:-}" ]] && PAIRS+=(B2_R:D_R)
  "${CLI[@]}" audit-depth --scene "$SCENE" --cameras "${CAM_LIST[@]}" --observations "${OBS_FILES[@]}" --config "$CONFIG" \
      --run "${RUNS[@]}" "${EXTRA[@]}" --pairs "${PAIRS[@]}" --tcp-npz "$ARCH" --output "$OUT/NEXT_A1_depth_audit_$TAG"
fi
if has A3; then
  POL=(--fold-from "$B2" --uncertainty cluster_robust --remote-hinge-policy observability)
  "${CLI[@]}" fit "${BASE[@]}" "${POL[@]}" --no-depth --experiment-id NEXT_A3_V2D --output "$OUT/NEXT_A3_V2D_$TAG"
  "${CLI[@]}" fit "${BASE[@]}" "${POL[@]}" --experiment-id NEXT_A3_V2D_rawD --output "$OUT/NEXT_A3_V2D_rawD_$TAG"
  "${CLI[@]}" fit "${BASE[@]}" "${POL[@]}" --depth-evidence evidence --tcp-npz "$ARCH" --experiment-id NEXT_A3_V2D_evidenceD \
      --output "$OUT/NEXT_A3_V2D_evidenceD_$TAG"
  "${CLI[@]}" fit "${BASE[@]}" "${POL[@]}" --depth-ar1 calibrate --experiment-id NEXT_A3_V2D_ar1D --output "$OUT/NEXT_A3_V2D_ar1D_$TAG"
  if (( ${#CAM_LIST[@]} > 1 )); then
    for i in "${!CAM_LIST[@]}"; do
      "${CLI[@]}" fit --scene "$SCENE" --config "$CONFIG" --cameras "${CAM_LIST[$i]}" --observations "${OBS_FILES[$i]}" \
          "${POL[@]}" --experiment-id "NEXT_A3_${CAM_LIST[$i]}_rawD" --output "$OUT/NEXT_A3_${CAM_LIST[$i]}_rawD"
    done
  fi
fi
if has C2; then
  "${CLI[@]}" folds --scene "$SCENE" --cameras "${CAM_LIST[@]}" --observations "${OBS_FILES[@]}" --config "$CONFIG" \
      --fold-from "$B2" --output "$OUT/NEXT_C2_folds_rawD_$TAG"
fi
if has D1; then
  D1="$OUT/NEXT_D1_contact_windows_$TAG"
  [[ -d "$D1" ]] || "$PY" samuele/audit_joint_contact_windows.py --scene "$SCENE" --clip-frames "$FRAMES" \
      --cameras "${CAM_LIST[@]}" --output "$D1"
fi
ls -d "$OUT"/NEXT_A3_*"$TAG" "$OUT"/NEXT_A0_* 2>/dev/null | grep -v '\.log$' | xargs -r "${CLI[@]}" summarize --runs
echo "outputs in $OUT"
