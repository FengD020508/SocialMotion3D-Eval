#!/usr/bin/env bash
#SBATCH --job-name=sm3d-droid13
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/zwang-24/shells/%x_%j.out
#SBATCH --error=/home/infres/zwang-24/shells/%x_%j.err

set -euo pipefail

: "${INPUT_ROOT:?export INPUT_ROOT}"
: "${CAMERA_ROOT:?export CAMERA_ROOT}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"

PROJECT_ROOT="/home/infres/zwang-24/LGM2Motion"
PYTHON_BIN="/home/infres/zwang-24/ENTER/envs/lgm2motion/bin/python"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl

for clip_id in {06..18}; do
    video="$(find "${INPUT_ROOT}" -maxdepth 1 -type f -name "${clip_id}_*.mp4" -print -quit)"
    if [[ -z "${video}" ]]; then
        echo "missing input for ${clip_id}" >&2
        exit 2
    fi
    scene="$(basename "${video}" .mp4)"
    camera_dir="${CAMERA_ROOT}/${scene}"
    mkdir -p "${camera_dir}"
    if [[ ! -s "${camera_dir}/camera_trajectory.npz" ]]; then
        "${PYTHON_BIN}" scripts/idd_ped/run_droid_slam.py \
            --inputs "${INPUT_ROOT}" \
            --manifest "${INPUT_ROOT}/selection_manifest.json" \
            --clip_id "${clip_id}" \
            --output "${camera_dir}/camera_trajectory.npz"
    fi
done

"${PYTHON_BIN}" scripts/idd_ped/run_targetlocked_pipeline.py \
    --inputs "${INPUT_ROOT}" \
    --source_outputs "${CAMERA_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --clips 06 07 08 09 10 11 12 13 14 15 16 17 18
