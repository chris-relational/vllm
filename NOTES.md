# TL; DR
I log here reading conprehension notes the repository. These include colateral nuances
not directly related to the task at hand but are required to accomplish it. For example, 
if a particular CPP-compilation or linking flag is required for build, I may add a note regarding the
flag in a general context.  

The `var` directory is used for files not meant to be uploaded to the repo.  
`var` is contained in `.gitignore` so it does not tamper with the repo code.  


# New Folders and Files
I create new folders and files containing additions I do while studying the repo.  
Ensure these folders and files do not meix with main and there are no local files or folders with
the same names (e.g. created while in the main branch).

## `notebooks` folder
It contains notebooks with inference code using HF LLMs, one notebook per LLM. Each such notebook contains vLLM client code, OpenAI REST API or HF transformers direct inference code.  
There's also a `huggingface-hub.ipyb` notebook with test code for the `huggingface-hub` library. This includes
the `huggingface.inference-client` library.  

The folder also contains `util.py` with common code for inference using `transformers`.  


# Building vLLM from sources on Apple Silicon (arm64+MacOS)
Follow the instruction [here](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html).  
Everything works without issues.  I have tested the host-installation with several models from HF.  

__Models:__  
1. meta-llama/Llama-3.2-1B   
2. meta-llama/Llama-3.2-1B-Instruct  
3. mistralai/mistral-7B-Instruct-v0.1  

## Caveat: `mistralai/Magistral-Small-2506
This model does not work (not only on vLLM; `transformers`-based direct inference does not work as well).  



# Building `vLLM` inside a docker container for ARM architecture (`docker/Dockerfile.arm`)

[The documentation]() provides the exact build commands for linux@x86. For Apple Silicon we must modify the build process a bit:
```shell
$ docker build -f docker/Dockerfile.cpu --tag vllm-cpu-env .
```

## Container `build` Comments
1. The dockerfile is `docker/Dockerfile.arm`.   

2. `--target vllm-openai` is particular to `Dockerfile.cpu` (it is a multitarget build and the `vllm-openai` target is only built).   

3. The command `ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4"`
   sets the container environment variable LD_PRELOAD to the reference shared library 
   (note this is an Ubuntu image created on an arm64 host).  

   LD_PRELOAD instructs the dynamic linker to load a shared library before any other library when running executables.  
   It allows you to override functions in system libraries or inject extra functionality without changing the application binary.  

   `tcmalloc_minimal` is the minimal version of Google’s TCMalloc library (Thread-Caching Malloc), 
   an optimized memory allocator from the "Google Performance Tools suite". It provides faster malloc/free 
   than the default system allocator (glibc malloc). Helps improve performance of memory-intensive apps.


```bash
# Launching OpenAI server 
docker run --rm -it \
   --privileged=true \
   --shm-size=4g \
   -p 8000:8000 \
   -v $HOME/.cache/huggingface:/huggingface \
   -e HF_HOME=/huggingface \
   -e VLLM_CPU_KVCACHE_SPACE=10 \
   -e VLLM_CPU_OMP_THREADS_BIND=0-4 \
   -e LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4" \
   --entrypoint bash \
   vllm-cpu-env \
   --model=meta-llama/Llama-3.2-1B-Instruct \
   --dtype=bfloat16
```

## Container `run` Comments
The environment variables used are explained in the same documentation page [right after the build command]
(https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html?h=#related-runtime-environment-variables).


# Additional installation requirements
1. For the chat REPL of `transformers` you need to `pip install accelerate`. This is a framework for distributing inference to
multiple nodes

2. To run the `generate` SDK it is recommended to `pip install bitsandbytes` (by Huggingface).  
   `bitsandbytes` has methods for quantizing (when loading to memory) LLMs that greately improves performance.  

