#!/usr/bin/env bash
set -euo pipefail

workspace_root="${SOCIALMOTION_WORKSPACE:-/workspace}"
genmo_root="${GENMO_ROOT:-${workspace_root}/LGM2Motion}"
# Environments contain many small files and are much faster on the Pod-local
# container disk. Inputs, checkpoints, caches and outputs remain on /workspace.
env_root="${SOCIALMOTION_ENV_ROOT:-/opt/socialmotion_envs}"
cache_root="${SOCIALMOTION_CACHE_ROOT:-${workspace_root}/cache}"

mkdir -p "${env_root}" "${cache_root}/uv" "${genmo_root}/third_party"
export UV_CACHE_DIR="${cache_root}/uv"

python -m pip install --quiet --upgrade uv
if [[ ! -x "${env_root}/gem/bin/python" ]]; then
  uv venv --python 3.10 --system-site-packages "${env_root}/gem"
fi
# The selected RunPod image already provides the verified CUDA 12.4 builds of
# torch 2.6 and torchvision 0.21. Do not let a generic resolver replace them
# with a newer CUDA major release.
"${env_root}/gem/bin/python" -m pip install --no-deps -e "${genmo_root}"
"${env_root}/gem/bin/python" -m pip install \
  'timm==0.6.7' 'lightning==2.3.0' 'hydra-core==1.3' hydra-zen hydra-colorlog \
  rich 'numpy==1.23.5' 'setuptools>=68.0' opencv-python ffmpeg-python \
  scikit-image termcolor einops 'imageio==2.34.1' 'av<14' joblib trimesh \
  open3d 'ultralytics==8.3.50' yacs smplx tqdm scipy pillow wandb \
  sentencepiece transformers huggingface_hub pytest ruff

if [[ ! -d "${genmo_root}/third_party/DROID-SLAM/.git" ]]; then
  git clone --recursive https://github.com/princeton-vl/DROID-SLAM.git \
    "${genmo_root}/third_party/DROID-SLAM"
fi
if [[ ! -d "${genmo_root}/third_party/mega-sam/.git" ]]; then
  git clone --recursive https://github.com/mega-sam/mega-sam.git \
    "${genmo_root}/third_party/mega-sam"
fi

"${env_root}/gem/bin/python" - <<'PY'
import torch
print({
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
})
PY
