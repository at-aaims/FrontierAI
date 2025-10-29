# FrontierAI
AI related notes on Frontier supercomputer at OLCF

# Contents

- Kernels
  * [FlashAttention](#flashattention) -- Junqi
    + [CK](#ck)
    + [Triton](#triton)
    + [SDPA](#sdpa) -- Aris
  * [GEMM](#gemm) -- Junqi
- Training
  * [PyTorch](#pytorch) -- Aris
    + [FSDP](#fsdp)
  * [DeepSpeed](#deepspeed) -- Emily
    + [MoE](#moe) -- Sajal
  * [Jax](#jax) -- Emily
  * [Megablock](#megablock) -- Sajal
  * [Verl](#verl) -- Vanessa
  * [X-MoE](#x-moe) -- Sajal
- Serving  -- Jesse, Junqi 
  * [vLLM](#vllm) 
  * [Ollama](#ollama)
  * [SGLang](#sglang)
- Agentic
  * [LangChain](#langchain) -- Emily
  * [AutoGen](#autogen) -- Vanessa


## FlashAttention 
### Installation 
FA2 is supported on Frontier and the upstream [repo](https://github.com/Dao-AILab/flash-attention) can be pip installed
```bash
module load PrgEnv-gnu
module load rocm
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh -b -p $WRKSPC/miniconda
export PATH=$WRKSPC/miniconda/bin:$PATH
conda create --prefix $WRKSPC/miniconda/envs/fa2-env -y
source $WRKSPC/miniconda/etc/profile.d/conda.sh
conda activate $WRKSPC/miniconda/envs/fa2-env
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
git checkout v2.8.3
pip install -e .
popd
```
For latest development, please try AMD's [fork](https://github.com/ROCm/flash-attention) 
### Backend
Standalone FA: 
- CK (default)
- Triton

PyTorch `scaled_dot_product_attention` (SDPA):
- Math
- FA
- Efficient 
### Performance 
- For standalone FA, use latest rocm. The built against rocm/6.3 is 1.5x faster than that with rocm/6.1 for certain inputs
  ![FA2](FlashAttention/fa2.png)

- For PyTorch SDPA, use FA or Efficient backend  
 ![SDPA](FlashAttention/sdpa.png)

## X-MoE
### Official X-MoE Code and Documentation

https://github.com/Supercomputing-System-AI-Lab/X-MoE

### Preparing Conda Environment

    module reset
    module load cpe/24.11
    module load PrgEnv-gnu/8.6.0
    module load rocm/6.3.1
    module load craype-accel-amd-gfx90a
    module load miniforge3/23.11.0-0


    # cd to your directory of choice,
    # I recommend using /lustre/orion directories as these packages require lots of space.

    conda create -p $PWD/TORCH-ROCM6.3.1_env python=3.11 -c conda-forge -y
    source activate $PWD/TORCH-ROCM6.3.1_env


### Installing Torch

    # ROCM 6.3.1
    pip3 install --pre torch torchvision torchaudio --index-url
    https://download.pytorch.org/whl/nightly/rocm6.3

### Installing APEX

    git clone https://github.com/ROCm/apex.git
    cd apex
    pip install -r requirements.txt
    python setup.py install --cpp_ext --cuda_ext

### Installing FlashAttention

Didn't work with pip install, so tried to install from source instead.

    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention/
    python setup.py install
    pytest -q -s tests/test_flash_attn.py

### Installing X-MoE

Worked as is.

    cd ~
    git clone https://github.com/Supercomputing-System-AI-Lab/X-MoE
    cd X-MoE
    git submodule update --init --recursive --remote

    pip install -e .
    cd Megatron-DeepSpeed-X-MoE && pip install -e .

### Data Preparation

Recommending removing   from XMoE's parent's directory. Worked as is.

Maybe make the amount of data a variable so that we don't have to go
through the entire dataset.

### Training on Single Node

Didn't work on login node. Looks like this depends on having a host
list.

    scontrol: error: host list is empty
    first=
    ssh: Could not resolve hostname : Name or service not known
    MASTER_ADDR=

### Getting an Interactive Node

        salloc -A PROJID -J RunSim123 -t 0:30:00 -p batch -N 1

### Running the Training

        ./X-MoE-Small-node-1.sh 8 1

Frontier pytorch $cpp\_extension$ related problem, \"ninja -v\" needs to
be replaced with

        ninja --version

.

Need to install mpi4py

        MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py

Replaced torchrun with srun

        time srun -u -N 1 -n${NUM_GPUS} -c2 --ntasks-per-node=8
        --gpus-per-node=8 --gpu-bind=closest python pretrain_gpt_deepspeed.py ...

### Error Logs

./X-MoE-Small-node-1-srun.sh 8 1

    srun_logs_with_jit_error.log

         ImportError: /lustre/orion/stf218/world-shared/sajal/testing_xmoe/
         X-MoE/Megatron-DeepSpeed-X-MoE/
         megatron/fused_kernels/build/scaled_upper_triang_masked_softmax_cuda.so:
         cannot open shared object file: No such file or directory

### Using Correct Versions of Compiler

        export CXX=/opt/cray/pe/gcc-native/13/bin/g++
        export CC=/opt/cray/pe/gcc-native/13/bin/gcc
        export PATH=/opt/cray/pe/gcc-native/13/bin:$PATH
