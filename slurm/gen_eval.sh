#!/bin/bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000245
#SBATCH --qos=highpriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x-%j.out
# Usage: NAME=<run name> SHARD=<k> NSHARDS=<n> bash slurm/gen_eval.sh [extra eval_shard args]
set -euo pipefail
cd /scratch/prj0000000245/sakurai/ncm
export HF_HOME=/scratch/prj0000000245/sakurai/hf_cache TOKENIZERS_PARALLELISM=false
CFG=config/eval/lm1b-small.yaml
OUT=outputs/partial_refinement/gen/${NAME}
mkdir -p $OUT
.venv/bin/python scripts/eval_shard.py --config $CFG --checkpoint-dir artifacts/lm1b/small \
  --shard ${SHARD:-0} --nshards ${NSHARDS:-1} --out $OUT/shard_${SHARD:-0}.pkl "$@"
