#!/usr/bin/env bash
#SBATCH --job-name=sm3d-e3-18
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/infres/zwang-24/shells/%x_%j.out
#SBATCH --error=/home/infres/zwang-24/shells/%x_%j.err

set -euo pipefail

: "${E3_CONFIG:?export E3_CONFIG}"

REPO_ROOT="/home/infres/zwang-24/LGM2Motion/SocialMotion3D-Eval"
PYTHON_BIN="/home/infres/zwang-24/ENTER/envs/lgm2motion/bin/python"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m unittest discover -s tests -v
"${PYTHON_BIN}" scripts/run_e3.py --config "${E3_CONFIG}"
