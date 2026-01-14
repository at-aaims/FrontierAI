# ROCm DeepSpeed SFT Setup (AMD GPUs)

This repository provides a reproducible setup for running **Supervised Fine-Tuning (SFT)** with **DeepSpeed** on ROCm-enabled AMD GPU systems. It covers environment creation, dependency installation, model and dataset downloads from Hugging Face, and launching an SFT job via Slurm.

The instructions are designed to work on HPC systems with module environments (e.g., Frontier-like systems), but can be adapted to other ROCm platforms.

---

## Requirements

* AMD GPUs with ROCm support (tested with ROCm 6.1.x)
* A system with environment modules (`module load`)
* Slurm workload manager
* A Hugging Face account and access token
* A writable filesystem for caching models and datasets

Define a writable root directory for environments and caches:

```bash
export SCRATCH_ROOT=/path/to/your/workspace
```

---

## Environment Setup

Load the required compiler, ROCm, and Conda tooling:

```bash
module reset
module load gcc/12.2.0
module load rocm/6.1.3
module load miniforge3/23.11.0
```

Initialize Conda and create an isolated environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p $SCRATCH_ROOT/envs/frontier-ft python=3.10
conda activate $SCRATCH_ROOT/envs/frontier-ft
module unload miniforge3/23.11.0
```

---

## PyTorch (ROCm) Installation

Install ROCm-compatible PyTorch wheels:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/rocm6.1 \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1
```

Install training and evaluation dependencies:

```bash
pip install \
  accelerate==1.10.1 \
  datasets==3.1.0 \
  evaluate==0.4.3 \
  peft==0.17.1 \
  safetensors==0.6.2 \
  sentencepiece==0.2.1 \
  tensorboard==2.20.0 \
  transformers==4.57.0 \
  trl==0.23.1
```

---

## DeepSpeed Installation

Clone and install DeepSpeed with fused Adam enabled:

```bash
git clone https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed
DS_BUILD_FUSED_ADAM=1 pip install -e .
```

---

## Hugging Face Setup

### Authentication

Export your Hugging Face token:

```bash
export HF_TOKEN=<your_huggingface_token>
```

### Cache Directories

Create cache directories for models, tokenizers, and datasets:

```bash
mkdir -p $SCRATCH_ROOT/llama-models-cache
mkdir -p $SCRATCH_ROOT/llama-tokenizers-cache
mkdir -p $SCRATCH_ROOT/data_cache/mlabonne___guanaco-llama2-1k
```

---

## Download Models and Data

### LLaMA 3 (8B) Model Weights

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --repo-type model \
  --local-dir $SCRATCH_ROOT/llama-models-cache \
  --local-dir-use-symlinks False
```

### LLaMA 3 Tokenizer

```bash
huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --repo-type tokenizer \
  --local-dir $SCRATCH_ROOT/llama-tokenizers-cache \
  --local-dir-use-symlinks False
```

### Guanaco Instruction-Tuning Dataset (1k)

```bash
huggingface-cli download mlabonne/guanaco-llama2-1k \
  --repo-type dataset \
  --local-dir $SCRATCH_ROOT/data_cache/mlabonne___guanaco-llama2-1k \
  --local-dir-use-symlinks False
```

---

## Running Supervised Fine-Tuning

Submit the DeepSpeed SFT job via Slurm:

```bash
sbatch run_sft.slurm
```

Ensure that `run_sft.slurm` specifies:

* The Conda environment path
* Model, tokenizer, and dataset cache locations
* DeepSpeed configuration (e.g., ZeRO stage, precision)
* Node and GPU counts appropriate for your system

---

## Notes on Portability

* Version pinning is intentional to ensure ROCm and DeepSpeed compatibility.
* All filesystem paths are externalized via `$SCRATCH_ROOT`.
* After initial download, Hugging Face assets can be reused offline.

