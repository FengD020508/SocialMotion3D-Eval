#!/usr/bin/env bash
set -euo pipefail

genmo_root="${GENMO_ROOT:-/workspace/LGM2Motion}"
python_bin="${GEM_PYTHON:-/opt/socialmotion_envs/gem/bin/python}"

download_gdrive() {
  local file_id="$1"
  local output="$2"
  if [[ -s "${output}" ]]; then
    return
  fi
  mkdir -p "$(dirname "${output}")"
  "${python_bin}" -m gdown "${file_id}" -O "${output}"
}

# File IDs are from the GVHMR Google Drive folder linked by GENMO's official
# docs/INSTALL.md (Steps 6 and 7).
download_gdrive 1X5hvVqvqI9tvjUCb2oAlZxtgIKD9kvsc \
  "${genmo_root}/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"
download_gdrive 1sR8xZD9wrZczdDVo6zKscNLwvarIRhP5 \
  "${genmo_root}/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"

(cd "${genmo_root}" && "${python_bin}" - <<'PY'
from gem.utils.hf_utils import download_checkpoint

print(download_checkpoint())
PY
)

stat -c '%n %s' \
  "${genmo_root}/inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt" \
  "${genmo_root}/inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"
