#!/bin/bash
#SBATCH --partition=testqueue
#SBATCH --account=prj0000000245
#SBATCH --qos=highpriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=slurm/%x-%j.out
set -uo pipefail
cd /scratch/prj0000000245/sakurai/ncm
export HF_HOME=/scratch/prj0000000245/sakurai/hf_cache
L48="4:{n},5:{n},6:{n},7:{n},8:{n}"
specs=(
 "P1|"
 "P1_T0.95|--temp 0.95"
 "P1_T0.9|--temp 0.9"
 "P1_T0.85|--temp 0.85"
 "P2_4-8|--passes ${L48//\{n\}/2}"
 "P4_4-8|--passes ${L48//\{n\}/4}"
 "P8_4-8|--passes ${L48//\{n\}/8}"
 "R4_4-8|--passes ${L48//\{n\}/4} --policy random"
 "L4_4-8|--passes ${L48//\{n\}/4} --policy l2r"
 "P4cos_4-8|--passes ${L48//\{n\}/4} --schedule cosine"
 "P4_1-10|--passes 1:4,2:4,3:4,4:4,5:4,6:4,7:4,8:4,9:4,10:4"
 "P2_1-11|--passes 1:2,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:2,10:2,11:2"
 "P4_11|--passes 11:4"
)
for spec in "${specs[@]}"; do
  NAME=${spec%%|*}; ARGS=${spec#*|}
  echo "=== $NAME [$ARGS]"
  NAME=$NAME SHARD=0 NSHARDS=1 bash slurm/gen_eval.sh $ARGS > outputs/partial_refinement/gen_${NAME}.log 2>&1 || { echo "FAILED $NAME"; continue; }
  .venv/bin/python -c "import pickle;d=pickle.load(open('outputs/partial_refinement/gen/$NAME/shard_0.pkl','rb'));print('gen_seconds %.2f samples %d'%(d['gen_seconds'],len(d['hyps'])))"
  .venv/bin/python scripts/eval_merge.py --config config/eval/lm1b-small.yaml --glob "outputs/partial_refinement/gen/$NAME/shard_*.pkl" 2>&1 | grep -v Warning | sed -n '/seed/,$p'
done
