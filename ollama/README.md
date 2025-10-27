# Ollama

ollama can be easily run using apptainer and the official ollama docker image.
The docker image is available here: https://hub.docker.com/r/ollama/ollama

Example:
```bash
apptainer build ./ollama.sif docker://ollama/ollama:0.12.5-rc0-rocm
apptainer exec ./ollama.sif ollama serve
```

You can use apptainer "instances" to manage running the ollama server in the background, which can
make setting up models easier:
```bash
apptainer instance start ./ollama.sif ollama
apptainer exec ./ollama.sif ollama serve
# In another shell (or send ollama serve to the background)
apptainer exec ./ollama.sif ollama run nemotron-mini:4b
```

## Internet
Note that by default Frontier job nodes don't have internet access. Either pre-fetch the models on
the login node with `ollama pull` or set up the internet proxy:

```bash
export https_proxy='http://proxy.ccs.ornl.gov:3128'
export http_proxy='http://proxy.ccs.ornl.gov:3128'
export no_proxy='localhost,127.0.0.1'
```
