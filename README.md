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
