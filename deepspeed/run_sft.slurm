#!/bin/bash 
#SBATCH -A stf218-arch 
#SBATCH -J sft-ds 
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1 # ONE task per node 
#SBATCH --gpus-per-node=4 
#SBATCH --cpus-per-task=288 # give torchrun enough CPUs 
#SBATCH -o logs/sft-ds-olcf-n2-%j.o 
#SBATCH -e logs/sft-ds-olcf-n2-%j.e 
#SBATCH --time=6:00:00 

set -euo pipefail 

module load miniforge3/24.11.3-2
module load nvhpc-hpcx-cuda12/24.3
module load PrgEnv-nvidia/8.5.0
module load cuda/12.9
source activate base
conda activate envs/frontier-ft

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-} # ensure recent toolchain is picked up when compiling DS ops 
export CC=$(which gcc)
export CXX=$(which g++)

# place DS build artifacts in scratch to avoid reusing stale system-wide binaries 
export TORCH_EXTENSIONS_DIR=$PWD/deepspeed_extensions
mkdir -p $PWD/deepspeed_extensions #"$TORCH_EXTENSIONS_DIR" 

export DS_ENV_FILE=$PWD/.deepspeed_env

# refresh DeepSpeed launcher environment so worker nodes inherit the right toolchains 
cat <<EOF > "$PWD/.deepspeed_env"
PATH=$PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH
TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR
CC=$CC
CXX=$CXX
HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1
TRANSFORMERS_OFFLINE=1
EOF

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Let Slurm/torchrun control device visibility/binding 
unset CUDA_VISIBLE_DEVICES # Networking 
export NCCL_SOCKET_IFNAME="^lo,docker,virbr"
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=$((64*1024*1024))

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901

scontrol show hostnames $SLURM_NODELIST > job.node.list.$SLURM_JOB_ID
input="./job.node.list".$SLURM_JOB_ID
host_file=host_file.$SLURM_JOB_ID
readarray -t arr <"$input"

for item in "${arr[@]}"; do
  echo "$item" slots=4 >> $host_file
done

# Multi-node DeepSpeed launch 
deepspeed --num_gpus 8 --num_nodes 2 --hostfile $host_file --master_addr=$MASTER_ADDR --master_port=29500 sft_llama_ds.py
