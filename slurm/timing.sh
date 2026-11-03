set -euo pipefail
cd /scratch/prj0000000245/sakurai/ncm
export HF_HOME=/scratch/prj0000000245/sakurai/hf_cache
for spec in "t_P1|" "t_P1via|--passes 4:1" "t_P4_4-8|--passes 4:4,5:4,6:4,7:4,8:4"; do
  NAME=${spec%%|*}; ARGS=${spec#*|}
  s=$(date +%s); NAME=$NAME SHARD=0 NSHARDS=5 bash slurm/gen_eval.sh $ARGS > /dev/null 2>&1; e=$(date +%s)
  .venv/bin/python -c "import pickle;d=pickle.load(open('outputs/partial_refinement/gen/$NAME/shard_0.pkl','rb'));print('$NAME', len(d['hyps']),'samples gen_s=%.1f'%d['gen_seconds'],'wall_s=$((e-s))')"
  s=$(date +%s); .venv/bin/python scripts/eval_merge.py --config config/eval/lm1b-small.yaml --glob "outputs/partial_refinement/gen/$NAME/shard_*.pkl" 2>&1 | grep -A2 genPPL; e=$(date +%s); echo "merge_s=$((e-s))"
done
.venv/bin/python -c "
import pickle;a=pickle.load(open('outputs/partial_refinement/gen/t_P1/shard_0.pkl','rb'));b=pickle.load(open('outputs/partial_refinement/gen/t_P1via/shard_0.pkl','rb'))
print('P1 == P1-via-refine raw ids:', a['raw_ids']==b['raw_ids'])"
