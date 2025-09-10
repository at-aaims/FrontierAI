
vLLM can only be used on Frontier either as:

* apptainer using docker image
* installation from source
* `pip install vllm` does not work

 Docker Hub ROCMm vllm:
 docker hub has the ROCm version 6.4.1 (https://hub.docker.com/r/rocm/vllm/tags)
 However when we actually see the version inside the image, it is still ROCm 6.3

Do not include the installation of any other package in the Docker image file, because it can override the initially installed packages and cause version compatibility issues.
    
library issue:
ROCm image was built on gdbv0.32, whereas as Frontier has gdbv0.35 or .37

apptainer command that worked for me:

"apptainer exec --fakeroot --writable-tmpfs ./vllm_rocm_64.sif"

It should include  both options; otherwise, it would give compatibility errors

No need to load ROCm modules 
