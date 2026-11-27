#!/usr/bin/env bash
# Model weights the ForceArtGS pipeline loads, into checkpoints/torch/hub/checkpoints/ (gitignored), sha256-checked.
# The pipeline copies them into each run's TORCH_HOME (pipelines/joint_aware_reconstruction.py: seed_model_weights);
# nothing is downloaded at run time.
#   alltracker.pth              AllTracker point tracker (joint discovery)      tracking/tracker_toolkit.py
#   convnext_tiny-983f1562.pth  AllTracker ConvNeXt backbone init (torchvision)
#   alexnet-owt-7be5be79.pth    LPIPS backbone, 3DGS evaluation (torchvision)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
D=checkpoints/torch/hub/checkpoints
mkdir -p "$D"
get() {  # name url sha256
  local f="$D/$1"
  if [[ ! -f "$f" ]] || ! echo "$3  $f" | sha256sum -c --quiet - 2>/dev/null; then
    curl -fL --retry 3 -o "$f.part" "$2" && mv "$f.part" "$f"
  fi
  echo "$3  $f" | sha256sum -c -
}
get alltracker.pth https://huggingface.co/aharley/alltracker/resolve/main/alltracker.pth \
    ffd9ebcfb6d206d594b646999a150540f92c049cf9b2bf940facf7123f62aa1d
get convnext_tiny-983f1562.pth https://download.pytorch.org/models/convnext_tiny-983f1562.pth \
    983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
get alexnet-owt-7be5be79.pth https://download.pytorch.org/models/alexnet-owt-7be5be79.pth \
    7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02
