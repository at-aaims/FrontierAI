Frontier Fine-Tuning Environment Setup (ROCm + DeepSpeed)

This repository documents the steps required to set up a ROCm-based PyTorch environment on Frontier (or similar AMD GPU systems), install DeepSpeed with fused optimizers, download required Hugging Face models and datasets, and run a supervised fine-tuning (SFT) example using DeepSpeed.

Prerequisites

Access to a Frontier-like HPC system with AMD GPUs

Environment modules available for gcc, rocm, and miniforge

A Hugging Face account and access token

$SCRATCH_ROOT environment variable set to a writable filesystem location

Example:

export SCRATCH_ROOT=/lustre/orion/<project>/<user>

Environment Setup

Reset modules and load the required toolchain:

module reset
module load gcc/12.2.0
module load rocm/6.1.3
module load miniforge3/23.11.0


Initialize Conda and create a dedicated environment:

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p $SCRATCH_ROOT/envs/frontier-ft python=3.10
conda activate $SCRATCH_ROOT/envs/frontier-ft
module unload miniforge3/23.11.0

PyTorch (ROCm) Installation

Install ROCm-compatible PyTorch and related libraries:

pip install --extra-index-url https://download.pytorch.org/whl/rocm6.1 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1


Install additional training and evaluation dependencies:

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

DeepSpeed Installation

Clone and install DeepSpeed with fused Adam enabled:

git clone https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed
DS_BUILD_FUSED_ADAM=1 pip install -e .

Hugging Face Setup

Export your Hugging Face access token:

export HF_TOKEN=<your_huggingface_token>


Create cache directories for models, tokenizers, and datasets:

mkdir -p $SCRATCH_ROOT/llama-models-cache
mkdir -p $SCRATCH_ROOT/llama-tokenizers-cache
mkdir -p $SCRATCH_ROOT/data_cache/mlabonne___guanaco-llama2-1k

Download Model Weights

Download the Meta LLaMA 3 8B model:

huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --repo-type model \
  --local-dir $SCRATCH_ROOT/llama-models-cache \
  --local-dir-use-symlinks False

Download Tokenizer
huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --repo-type tokenizer \
  --local-dir $SCRATCH_ROOT/llama-tokenizers-cache \
  --local-dir-use-symlinks False

Download Dataset

Download the Guanaco 1k instruction-tuning dataset:

huggingface-cli download mlabonne/guanaco-llama2-1k \
  --repo-type dataset \
  --local-dir $SCRATCH_ROOT/data_cache/mlabonne___guanaco-llama2-1k \
  --local-dir-use-symlinks False

Running Supervised Fine-Tuning (SFT)

Submit the DeepSpeed SFT job using Slurm:

sbatch run_sft.slurm


Ensure that run_sft.slurm is configured with the correct paths to:

The Conda environment

Model and tokenizer caches

Dataset cache

Desired DeepSpeed configuration (e.g., ZeRO stage, precision)

Notes

This setup is tailored for ROCm 6.1.x and AMD Instinct GPUs.

Version pinning is intentional to ensure compatibility with DeepSpeed fused optimizers.

For multi-node runs, confirm that your Slurm script correctly sets MASTER_ADDR, MASTER_PORT, and WORLD_SIZE.

If you want, I can also generate:

A run_sft.slurm template aligned with Frontier best practices

A minimal DeepSpeed JSON config (ZeRO-2 / ZeRO-3)

A version of this README tailored for inclusion directly in a public repo

A version of this README tailored for inclusion directly in a public repo


Below is a public-repo–ready README: neutral wording, no site-specific assumptions beyond “ROCm-based AMD GPU systems,” no private paths, and no internal jargon. It is suitable to place directly at the root of a GitHub repository.

ROCm DeepSpeed SFT Setup (AMD GPUs)

This repository provides a reproducible setup for running Supervised Fine-Tuning (SFT) with DeepSpeed on ROCm-enabled AMD GPU systems. It covers environment creation, dependency installation, model and dataset downloads from Hugging Face, and launching an SFT job via Slurm.

The instructions are designed to work on HPC systems with module environments (e.g., Frontier-like systems), but can be adapted to other ROCm platforms.

Requirements

AMD GPUs with ROCm support (tested with ROCm 6.1.x)

A system with environment modules (module load)

Slurm workload manager

A Hugging Face account and access token

A writable filesystem for caching models and datasets

You must define a writable root directory for environments and caches:

export SCRATCH_ROOT=/path/to/your/workspace

Environment Setup

Load the required compiler, ROCm, and Conda tooling:

module reset
module load gcc/12.2.0
module load rocm/6.1.3
module load miniforge3/23.11.0


Initialize Conda and create an isolated environment:

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -p $SCRATCH_ROOT/envs/frontier-ft python=3.10
conda activate $SCRATCH_ROOT/envs/frontier-ft
module unload miniforge3/23.11.0

PyTorch (ROCm) Installation

Install ROCm-compatible PyTorch wheels:

pip install --extra-index-url https://download.pytorch.org/whl/rocm6.1 \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1


Install training and evaluation dependencies:

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

DeepSpeed Installation

Clone and install DeepSpeed with fused Adam enabled:

git clone https://github.com/deepspeedai/DeepSpeed.git
cd DeepSpeed
DS_BUILD_FUSED_ADAM=1 pip install -e .

Hugging Face Authentication and Caching

Export your Hugging Face token:

export HF_TOKEN=<your_huggingface_token>


Create cache directories for models, tokenizers, and datasets:

mkdir -p $SCRATCH_ROOT/llama-models-cache
mkdir -p $SCRATCH_ROOT/llama-tokenizers-cache
mkdir -p $SCRATCH_ROOT/data_cache/mlabonne___guanaco-llama2-1k

Download Models and Data
LLaMA 3 (8B) Model Weights
huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --repo-type model \
  --local-dir $SCRATCH_ROOT/llama-models-cache \
  --local-dir-use-symlinks False

LLaMA 3 Tokenizer
huggingface-cli download meta-llama/Meta-Llama-3-8B \
  --repo-type tokenizer \
  --local-dir $SCRATCH_ROOT/llama-tokenizers-cache \
  --local-dir-use-symlinks False

Guanaco Instruction-Tuning Dataset (1k)
huggingface-cli download mlabonne/guanaco-llama2-1k \
  --repo-type dataset \
  --local-dir $SCRATCH_ROOT/data_cache/mlabonne___guanaco-llama2-1k \
  --local-dir-use-symlinks False

Running Supervised Fine-Tuning

To launch the DeepSpeed SFT example using Slurm:

sbatch run_sft.slurm


Ensure that run_sft.slurm correctly specifies:

The Conda environment path

Model, tokenizer, and dataset cache locations

DeepSpeed configuration (e.g., ZeRO stage, precision)

Node and GPU counts appropriate for your system

Notes on Portability

Version pinning is intentional to ensure ROCm and DeepSpeed compatibility.

Paths are externalized via $SCRATCH_ROOT to avoid hard-coded system dependencies.

This setup assumes offline or cached Hugging Face usage after initial download.
