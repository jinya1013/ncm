#!/bin/bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000245
#SBATCH --qos=highpriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/%x-%j.out
set -uo pipefail
cd /scratch/prj0000000245/sakurai/ncm
export HF_HOME=/scratch/prj0000000245/sakurai/hf_cache
L48="4:{n},5:{n},6:{n},7:{n},8:{n}"
specs=(
 "P1_T0.8|--temp 0.8"
 "P1_T0.75|--temp 0.75"
 "P1_T0.7|--temp 0.7"
)
for spec in "${specs[@]}"; do
  NAME=${spec%%|*}; ARGS=${spec#*|}
  echo "=== $NAME [$ARGS]"
  NAME=$NAME SHARD=0 NSHARDS=1 bash slurm/gen_eval.sh $ARGS > outputs/partial_refinement/gen_${NAME}.log 2>&1 || { echo "FAILED $NAME"; continue; }
  .venv/bin/python -c "import pickle;d=pickle.load(open('outputs/partial_refinement/gen/$NAME/shard_0.pkl','rb'));print('gen_seconds %.2f samples %d'%(d['gen_seconds'],len(d['hyps'])))"
  .venv/bin/python scripts/eval_merge.py --config config/eval/lm1b-small.yaml --glob "outputs/partial_refinement/gen/$NAME/shard_*.pkl" 2>&1 | grep -v Warning | sed -n '/seed/,$p'
done
