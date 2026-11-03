#!/bin/bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000245
#SBATCH --qos=highpriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
set -euo pipefail
cd /scratch/prj0000000245/sakurai/ncm
export HF_HOME=/scratch/prj0000000245/sakurai/hf_cache
N=${N:-2048}
.venv/bin/python scripts/eval_partial_refinement.py --config config/eval/lm1b-small.yaml \
  --checkpoint-dir artifacts/lm1b/small \
  --data /scratch/prj0000000245/sakurai/data/lm1b/lm1b_test_bs128_wrapped.dat \
  --n $N --out outputs/partial_refinement/oracle_lm1b_small_n${N}.json
