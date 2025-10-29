#!/usr/bin/env bash
# Single-node validation runner (no internet) for Frontier.

#!/bin/bash
#SBATCH -A stf218
#SBATCH -J lc
#SBATCH -o lc.%J.out
#SBATCH -e lc.%J.err
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:20:00
#SBATCH -p batch
#SBATCH -N 1

set -euo pipefail

export LLM_MODEL_DIR=".cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/8afb486c1db24fe5011ec46dfbe5b5dccdb575c2/"   # local HF model dir
export EMB_MODEL_DIR=".cache/huggingface/hub/models--intfloat--e5-base/snapshots/b533fe4636f4a2507c08ddab40644d20b0006d6a/"               # local embedding model dir (optional)
export DOCS_DIR="$PWD"                          # small local doc set

# Offline flags (defensive—script also sets these in Python)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module load rocm/6.4.0
source /lustre/orion/stf218/scratch/1eh/miniconda3/bin/activate
conda activate /lustre/orion/stf218/scratch/1eh/langchain_test/lc_env


HOME=$PWD python langchain_frontier_toy.py \
  --llm_model_dir "${LLM_MODEL_DIR}" \
  --emb_model_dir "${EMB_MODEL_DIR}" \
  --docs_dir "${DOCS_DIR}" \
  --faiss_index_dir "./toy_index" \
  --device auto \
  --dtype bf16 \
  --max_new_tokens 256 \
  --temperature 0.2 \
  --top_k 4
