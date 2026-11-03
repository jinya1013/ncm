#!/bin/bash
set -euo pipefail
cd /scratch/prj0000000245/sakurai/ncm
export UV_CACHE_DIR=/scratch/prj0000000245/sakurai/uv-cache HF_HUB_CACHE=/scratch/prj0000000245/sakurai/hf_cache/hub HF_HOME=/scratch/prj0000000245/sakurai/hf_cache
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python "torch==2.7.1" numpy omegaconf hydra-core "transformers>=4.51" "huggingface_hub[cli]" regex rouge_score sacrebleu datasets pytest ruff
.venv/bin/hf download hazyresearch/ncm-checkpoints --include "lm1b/tokenizer.pt" "lm1b/small/*" "lm1b/README.md" --local-dir artifacts
.venv/bin/hf download hazyresearch/ncm-datasets --repo-type dataset --include "lm1b/valid_*" --local-dir data
.venv/bin/python -c "import torch,transformers,datasets;print(torch.__version__,transformers.__version__,datasets.__version__)"
ls -laR artifacts data
