#!/usr/bin/env bash
#SBATCH --job-name=socialmotion3d-pilot
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR}"
PYTHON_BIN="/home/infres/zwang-24/ENTER/envs/lgm2motion/bin/python"

if [[ -z "${E3_CONFIG:-}" || -z "${E2A_CONFIG:-}" ]]; then
  echo "E3_CONFIG and E2A_CONFIG must be exported when submitting the job" >&2
  exit 2
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m unittest discover -s "${REPO_ROOT}/tests" -v
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_e3.py" --config "${E3_CONFIG}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/run_e2a.py" --config "${E2A_CONFIG}"

