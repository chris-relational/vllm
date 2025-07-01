# TL; DR
I log here reading conprehension notes for the repository. These may also include nuances
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



<!-- V L L M  B u i l d  o n  A R M 6 4  H o s t -->
# Building `vLLM` from sources on Apple Silicon (arm64+MacOS)
Follow the instruction [here](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html).  
Everything works without issues.  I have tested the host-installation with several models from HF.  

__Models:__  
1. meta-llama/Llama-3.2-1B   
2. meta-llama/Llama-3.2-1B-Instruct  
3. mistralai/mistral-7B-Instruct-v0.1  
4. mistralai/Magistral-Small-2506  
   <span style="color: red;">
   __CAVEAT!__ This model does not work (not only on vLLM; `transformers`-based  
   direct inference does not work as well)
   </span>     



<!-- B u i l d i n g  v L L M  f r o m   s o u r c e s  o n   L i n u x  x 8 6 -->
# Building `vLLM` from sources on Linux x86
<span style="color: red;">
Probably vLLM cannot be built on x86 Linux with CPU only. Eventually, it asks for CUDA features 
(see iten5 below)
</span>

1. Install gcc AND g++ (they're separate installations)
2. VLLM_TARGET_DEVICE=cpu (use everywhere, doesn't harm)
3. sudo apt isntall python3.12-dev (otherwise you get the following error that is knd of inexplicable:
   -- Could NOT find Python (missing: Python_INCLUDE_DIRS Interpreter Development.Module Development.SABIModule)
      CMake Error at cmake/utils.cmake:10 (message):
      Unable to find python matching:
      /home/ubuntu/projects/remotes/chris-relational/vllm/.venv/bin/python3.12.
      Call Stack (most recent call first):
      CMakeLists.txt:56 (find_python_from_executable)
  )
4. vLLM CMake uses ninja and NOT GMake as the backend build system. 
   Install ninja:
   $ sudo apt-get update && sudo apt-get install ninja-build

5. Eventual Command
   $ cmake /home/ubuntu/projects/remotes/chris-relational/vllm 
      '-G', 'Ninja', 
      '-DCMAKE_BUILD_TYPE=RelWithDebInfo', 
      '-DVLLM_TARGET_DEVICE=cpu', 
      '-DVLLM_PYTHON_EXECUTABLE=/home/ubuntu/projects/remotes/chris-relational/vllm/.venv/bin/python3.12', 
      '-DVLLM_PYTHON_PATH=/tmp/pip-build-env-khmpa04v/site:/usr/lib/python312.zip:/usr/lib/python3.12:/usr/lib/python3.12/lib-dynload:/tmp/pip-build-env-khmpa04v/overlay/lib/python3.12/
         site-packages:/tmp/pip-build-env-khmpa04v/normal/lib/python3.12/site-packages:/tmp/pip-build-env-khmpa04v/overlay/lib/python3.12/site-packages/setuptools/_vendor', 
      '-DFETCHCONTENT_BASE_DIR=/home/ubuntu/projects/remotes/chris-relational/vllm/.deps', 
      '-DCMAKE_JOB_POOL_COMPILE:STRING=compile', 
      '-DCMAKE_JOB_POOLS:STRING=compile=4'

   CUDA-related error (why?)
   CMake Error at /tmp/pip-build-env-jstth2gb/overlay/lib/python3.12/site-packages/torch/share/cmake/Caffe2/Caffe2Config.cmake:90 (message):
      Your installed Caffe2 version uses CUDA but I cannot find the CUDA
      libraries.  Please set the proper CUDA prefixes and / or install CUDA.
   Call Stack (most recent call first):
      /tmp/pip-build-env-jstth2gb/overlay/lib/python3.12/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:68 (find_package)
      CMakeLists.txt:80 (find_package).  
   <span style="color: red;">
   I filed the issue to vLLM:
   <span style="color: red;">
   `https://github.com/vllm-project/vllm/issues/20326`
   </span>



<!-- V L L M  B u i l d  o n  A R M 6 4  D o c k e r -->
# Building `vLLM` inside a docker container for ARM architecture (`docker/Dockerfile.arm`)

[The documentation](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html#build-image-from-source) 
provides the exact build commands for linux@x86. For Apple Silicon we must modify the build process a bit:
```shell
tag=cpu branch=playgound 
$ docker build -f docker/Dockerfile.cpu --tag vllm-cpu-env .


tag="arm" platform="linux/arm64/v8" \
target="cpu" repo=vllm branch=main \
bash -c '
docker build\
    --platform ${platform} \
    --build-arg BASE_NAME=${target} \
    --build-arg VLLM_TARGET_DEVICE=${target} \
    -f docker/Dockerfile.cpu \
    -t ${repo}-${branch}:${target}-${tag} \
    --target vllm-openai \
    --shm-size=8g .
'
```

## Comments
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



<!-- R u n n i n g  t h e  w e b - s e r v e r  f r o m  t h e  c o n t a i n e r  o n  A R M -->
# Running the vLLM OpenAI API web server from the container
```bash
# Launching OpenAI server 
docker run --rm -it \   
   -p 8000:8000 \
   -v $HOME/.cache/huggingface:/huggingface \
   -e HF_HOME=/huggingface \
   -e VLLM_CPU_KVCACHE_SPACE=4 \
   -e VLLM_CPU_OMP_THREADS_BIND=0-4 \
   -e MAX_MODEL_LENGTH=8192 \
   -e VLLM_LOGLEVEL=DEBUG \
   --entrypoint /bin/bash \
   vllm-cpu-env
```

## Comments `2025-06-29`
The image does not run on M3 Mac. The following command fails to start the service and errs with error:  
```
libnuma: Warning: node argument -1 is out of range
get_mempolicy: Operation not permitted
```

## Container `run` Comments
The environment variables used are explained in the same documentation page [right after the build command]
(https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html?h=#related-runtime-environment-variables).



# Huggingface `transformers` 
1. For the chat REPL of `transformers` you need to `pip install accelerate`. This is a framework for distributing inference to
multiple nodes

2. To run the `transformers.generate` API it is recommended to `pip install bitsandbytes` (by Huggingface).  
   `bitsandbytes` has methods for quantizing (when loading to memory) LLMs that greately improves performance.  

