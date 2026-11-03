#!/bin/bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000245
#SBATCH --qos=highpriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/%x-%j.out
set -euo pipefail
cd /scratch/prj0000000245/sakurai/ncm
export HF_HOME=/scratch/prj0000000245/sakurai/hf_cache
.venv/bin/python -m pytest -x -q -s tests/test_partial_refinement.py "$@"
