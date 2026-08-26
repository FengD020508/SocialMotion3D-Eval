#!/usr/bin/env bash
#SBATCH --job-name=sm3d-mega13
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/infres/zwang-24/shells/%x_%j.out
#SBATCH --error=/home/infres/zwang-24/shells/%x_%j.err

set -euo pipefail

: "${INPUT_ROOT:?export INPUT_ROOT}"
: "${OUTPUT_ROOT:?export OUTPUT_ROOT}"

PROJECT_ROOT="/home/infres/zwang-24/LGM2Motion"
PYTHON_BIN="/home/infres/zwang-24/ENTER/envs/lgm2motion/bin/python"
MANAGED_JOB_TMP=0

if [[ -z "${SLURM_TMPDIR:-}" ]]; then
    export SLURM_TMPDIR="/tmp/zwang-24-sm3d-${SLURM_JOB_ID}"
    MANAGED_JOB_TMP=1
fi
case "${SLURM_TMPDIR}" in
    /tmp/zwang-24-sm3d-[0-9]*|/local/*|/scratch/*) ;;
    *) echo "unsafe job-local temporary path: ${SLURM_TMPDIR}" >&2; exit 2 ;;
esac
mkdir -p "${SLURM_TMPDIR}"

cleanup_job_tmp() {
    if [[ "${MANAGED_JOB_TMP}" -eq 1 && "${SLURM_TMPDIR}" == "/tmp/zwang-24-sm3d-${SLURM_JOB_ID}" ]]; then
        rm -rf -- "${SLURM_TMPDIR}"
    fi
}
trap cleanup_job_tmp EXIT

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=0
export PYOPENGL_PLATFORM=egl

"${PYTHON_BIN}" scripts/idd_ped/run_megasam_targetlocked_pipeline.py \
    --inputs "${INPUT_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --clips 06 07 08 09 10 11 12 13 14 15 16 17 18 \
    --resume \
    --cache_policy metric_tmp
