#!/usr/bin/env bash
set -euo pipefail

genmo_root="${GENMO_ROOT:-/workspace/LGM2Motion}"
mega_root="${genmo_root}/third_party/mega-sam"
env_prefix="${MEGASAM_ENV_PREFIX:-/opt/socialmotion_envs/mega_sam}"
miniforge_root="${MINIFORGE_ROOT:-/opt/miniforge}"
conda_bin="${CONDA_BIN:-${miniforge_root}/bin/conda}"
cache_root="${SOCIALMOTION_CACHE_ROOT:-/workspace/cache}"

if [[ ! -d "${mega_root}/.git" ]]; then
  echo "MegaSAM checkout is missing: ${mega_root}" >&2
  exit 1
fi

patch_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/patches/mega-sam-disable-focal-optimization.patch"
if git -C "${mega_root}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
  echo "MegaSAM focal-fallback patch is already applied"
elif git -C "${mega_root}" apply --check "${patch_file}"; then
  git -C "${mega_root}" apply "${patch_file}"
else
  echo "MegaSAM checkout is incompatible with ${patch_file}" >&2
  exit 1
fi

if [[ ! -x "${conda_bin}" ]]; then
  installer="${cache_root}/Miniforge3-Linux-x86_64.sh"
  mkdir -p "${cache_root}"
  if [[ ! -s "${installer}" ]]; then
    curl --fail --location --retry 3 \
      'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh' \
      --output "${installer}"
  fi
  bash "${installer}" -b -p "${miniforge_root}"
fi

if [[ ! -x "${env_prefix}/bin/python" ]]; then
  # Reproduce MegaSAM's official Python/CUDA/PyTorch pins, but install the pip
  # section separately.  Recent pip build isolation otherwise hides torch while
  # it prepares torch-scatter and aborts the complete conda transaction.
  "${conda_bin}" create -y -p "${env_prefix}" \
    -c pytorch -c nvidia -c conda-forge \
    python=3.10 pip cudatoolkit=11.8 pytorch=2.0.1 torchvision=0.15.2
fi

# PyTorch 2.0.1 can fail to import with the iJIT symbol exposed by newer MKL.
# Use the immutable conda-forge artifact directly: a full re-solve can fail
# when an old cudatoolkit build has rolled out of current repodata.
"${conda_bin}" install -y --no-deps -p "${env_prefix}" \
  'https://conda.anaconda.org/conda-forge/linux-64/mkl-2024.0.0-ha957f24_49657.conda'
"${conda_bin}" install -y --no-deps -p "${env_prefix}" \
  'https://conda.anaconda.org/nvidia/linux-64/cuda-nvcc-11.8.89-0.tar.bz2'
"${conda_bin}" install -y --no-deps -p "${env_prefix}" \
  'https://conda.anaconda.org/nvidia/linux-64/cuda-cccl-11.8.89-0.tar.bz2' \
  'https://conda.anaconda.org/nvidia/linux-64/cuda-cudart-dev-11.8.89-0.tar.bz2' \
  'https://conda.anaconda.org/nvidia/linux-64/libcublas-dev-11.11.3.6-0.tar.bz2' \
  'https://conda.anaconda.org/nvidia/linux-64/libcusparse-dev-11.7.5.86-0.tar.bz2' \
  'https://conda.anaconda.org/nvidia/linux-64/libcusolver-dev-11.4.1.48-0.tar.bz2'

export CUDA_HOME="${env_prefix}"
export PATH="${CUDA_HOME}/bin:${PATH}"

"${env_prefix}/bin/python" -m pip install setuptools==69.5.1
"${env_prefix}/bin/python" -m pip install \
  opencv-python-headless==4.9.0.80 \
  tqdm==4.67.1 imageio==2.36.0 einops==0.8.0 scipy==1.14.1 \
  matplotlib==3.9.2 wandb==0.18.7 timm==1.0.7 ninja==1.11.1 \
  numpy==1.26.3 huggingface-hub==0.23.4 kornia==0.7.4
"${env_prefix}/bin/python" -m pip install --no-index torch-scatter \
  -f 'https://data.pyg.org/whl/torch-2.0.1+cu118.html'

xformers_archive="${cache_root}/xformers-0.0.22.post7-py310-cu118-pyt201.tar.bz2"
mkdir -p "${cache_root}"
if [[ ! -s "${xformers_archive}" ]]; then
  curl --fail --location --retry 3 \
    'https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2' \
    --output "${xformers_archive}"
fi
"${conda_bin}" install -y -p "${env_prefix}" "${xformers_archive}"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"
(cd "${mega_root}/base" && "${env_prefix}/bin/python" setup.py install)

depth_dir="${mega_root}/Depth-Anything/checkpoints"
mkdir -p "${depth_dir}"
if [[ ! -s "${depth_dir}/depth_anything_vitl14.pth" ]]; then
  curl --fail --location --retry 3 \
    'https://huggingface.co/spaces/LiheYoung/Depth-Anything/resolve/main/checkpoints/depth_anything_vitl14.pth' \
    --output "${depth_dir}/depth_anything_vitl14.pth"
fi

"${env_prefix}/bin/python" - <<'PY'
import torch
import xformers
import lietorch
print({
    "torch": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "xformers": xformers.__version__,
    "lietorch": lietorch.__file__,
})
PY
