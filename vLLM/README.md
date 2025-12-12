# vLLM

vLLM can only be used on Frontier either as:

* apptainer using docker image
* installation from source

`pip install vllm` does *not* work.

Installing from source is quite slow and convoluted, using the docker image is recommended.
Docker images are available here: https://hub.docker.com/r/rocm/vllm/tags
The docker image contains ROCm, so no need to load the ROCm modules.

Example usage:
```bash
apptainer build ./vllm.sif docker://rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909
apptainer exec ./vllm.sif vllm serve nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1 --tp 1
```

## Internet
Note that by default Frontier job nodes don't have internet access, which can cause issues when vLLM
tries to fetch huggingface models. Either pre-fetch the models on the login node with `hf download`
or set up the internet proxy:

```bash
export https_proxy='http://proxy.ccs.ornl.gov:3128'
export http_proxy='http://proxy.ccs.ornl.gov:3128'
export no_proxy='localhost,127.0.0.1'
```

## Other issues
Frontier's MI250 GPUs do not support FP4 or FP8  which can cause some models to take much more ram or have issues running. On recent versions of vLLM FP4 quantized models like gpt-oss will fall back to FP16 properly.
See
- FP8, no support, https://docs.vllm.ai/en/latest/features/quantization/index.html
- FP4, no support, https://docJs.vllm.ai/en/latest/features/quantization/index.html
