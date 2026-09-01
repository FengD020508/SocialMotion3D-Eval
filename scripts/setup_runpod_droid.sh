#!/usr/bin/env bash
set -euo pipefail

genmo_root="${GENMO_ROOT:-/workspace/LGM2Motion}"
python_bin="${GEM_PYTHON:-/opt/socialmotion_envs/gem/bin/python}"
droid_root="${genmo_root}/third_party/DROID-SLAM"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export MAX_JOBS="${MAX_JOBS:-8}"

if [[ ! -d "${droid_root}/.git" ]]; then
  echo "DROID-SLAM checkout is missing: ${droid_root}" >&2
  exit 1
fi

"${python_bin}" -m pip install evo gdown tensorboard pyyaml

if ! "${python_bin}" - <<'PY'
import sys
from pathlib import Path
import torch

root = Path("/workspace/LGM2Motion/third_party/DROID-SLAM")
sys.path.insert(0, str(root / "droid_slam"))
import droid_backends  # noqa: F401
import lietorch  # noqa: F401
import torch_scatter  # noqa: F401
PY
then
  (cd "${droid_root}/thirdparty/lietorch" && "${python_bin}" setup.py install)
  (cd "${droid_root}/thirdparty/pytorch_scatter" && \
    FORCE_ONLY_CUDA=1 "${python_bin}" setup.py install)
  (cd "${droid_root}" && "${python_bin}" setup.py install)
fi

weights_dir="${genmo_root}/inputs/checkpoints/droid"
mkdir -p "${weights_dir}"
if [[ ! -s "${weights_dir}/droid.pth" ]]; then
  # Official DROID-SLAM tools/download_model.sh Google Drive file id.
  "${python_bin}" -m gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh \
    -O "${weights_dir}/droid.pth"
fi

"${python_bin}" - <<'PY'
import sys
from pathlib import Path

root = Path("/workspace/LGM2Motion/third_party/DROID-SLAM")
sys.path.insert(0, str(root / "droid_slam"))
import torch
import droid_backends  # noqa: F401
import lietorch
import torch_scatter
print({"droid_backends": "ok", "lietorch": lietorch.__file__, "torch_scatter": torch_scatter.__file__})
PY
