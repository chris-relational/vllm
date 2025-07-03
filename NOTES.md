# TL; DR
I log here reading conprehension notes from the repository. They may include nuances not directly related to the task at hand. 
For example, if a particular cpp-compilation or linking flag is required for build, I may add a note regarding the
general use of the flag.  



# Repository File and Folder Additions
I create new folders and files containing additions I do while studying the repo.  
Ensure these folders and files do not mess with main and there are no local files or folders with
the same names (e.g. created while in the main branch).


## `var` directory
It is used for files not meant to be uploaded to the repo.  
`var` is contained in `.gitignore` of the main branch so it does not mess with repo code.  


## `notebooks` directory
It contains notebooks with inference code using HF LLMs, one notebook per LLM. Each such notebook contains vLLM client code, OpenAI REST API or HF transformers direct inference code.  
There's also a `huggingface-hub.ipyb` notebook with test code for the `huggingface-hub` library. This includes
the `huggingface.inference-client` library.  

The folder also contains `util.py` with common code for inference using `transformers`.  



<!-- 
v L L M  B u i l d  a n d  D e p l o y m e n t  T e s t s 
. . . .  . . . . .  . . .  . . . . . . . . . .  . . . . .
-->
# vLLM Build and Deployment Tests

All the tests below are conducted on the following LLMs (downloaded from Huggingface):

| Model                                  | Test                 | References-Comments                                                                                                |
|----------------------------------------|----------------------|--------------------------------------------------------------------------------------------------------------------|
| mistralai/Magistral-Small-2506         | vLLM on host         | **Failure**  <br> Followed this playbook from the HF model card in a dedicated testing environment                 |
|                                        |                      | (`~/projects/local/playground/magistral-vllm`).  <br> Failed to run the vLLM server.                               |
| mistralai/Mistral-7B-Instruct-v0.1     | vLLM on host         | **Success**  <br> Ran `vllm serve mistralai/Mistral-7B-Instruct-v0.1`                                              |
|                                        |                      | in `~/projects/local/playground/magistral-vllm` and sent a POST using the CURL command in the model card.          |
| mistralai/Mistral-7B-Instruct-v0.1     | Transformers         | **Success**  <br> The model card on HF implies that the model doesn't have an AutoTokenizer-compatible             |
|                                        |                      | tokenizer and we have to use the `mistral_common` Pylib for tokenization, with `AutoModelForCausalLM`              |
|                                        |                      | for inference.  <br> This is not true. The `README.md` in the `snapshots` folder of the model contains             |
|                                        |                      | an example with `AutoTokenizer`.  <br> The inference is in `~/projects/remotes/public/vllm/var/mistralai.ipynb`.   |
| meta-llama/Llama-3.2-3B-Instruct       | vLLM on host         | **Success**  <br> NOTE: must set `--max-model-len` to a small value e.g. `4096`.                                   |
| meta-llama/Llama-3.2-3B-Instruct       | Transformers on host | *(No comments provided)*                                                                                           |
| meta-llama/Llama-3.2-1B-Instruct       | vLLM on host         | **SUCCESS**                                                                                                        |



<!-- V L L M  S o u r c e  B u i l d  o n  A R M 6 4 -->
## `vLLM` Source Build on MacOS/arm64 (M3)
__SUCCESS__  
We followed the instructions [in the vLLM web site](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html).  
We tested the host-installation with the following models from HF:  

1. `meta-llama/Llama-3.2-1B`   
2. `meta-llama/Llama-3.2-1B-Instruct`  
3. `mistralai/mistral-7B-Instruct-v0.1`  

4. `mistralai/Magistral-Small-2506`  
   <span style="color: red;">
   __CAVEAT!__ This model does not work (not only on vLLM; `transformers`-based  
   direct inference does not work as well)
   </span>     



<!-- V L L M  S o u r c e  B u i l d  o n  L i n u x  x 8 6 -->
# `vLLM` Source Build on Linux/amd64 (i9)
__FAILURE__
<span style="color: red;">
Probably vLLM cannot be built on x86 Linux with CPU only. Eventually, it asks for CUDA features 
(see iten5 below).  
We filed a [ticket at the vLLM repo](https://github.com/vllm-project/vllm/issues/20326).  
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
# `vLLM` container build and run for ARM architecture (`docker/Dockerfile.arm`)

## Build
__SUCCESS__
[The documentation](https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html#build-image-from-source) 
provides the exact build commands for linux@x86. For Apple Silicon we must modify the build process a bit:
```shell
tag=cpu branch=playgound platform=arm64 \
bash -c '
   docker build \
   -f docker/Dockerfile.cpu \
   -t vllm-${branch}:${tag}-${platform} .
'
```

### Comments
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


## Run
__FAILURE__

The environment variables used are explained in the same documentation page [right after the build command]
(https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html?h=#related-runtime-environment-variables).

```bash
tag=cpu branch=playgound platform=arm64 \
bash -c '
   docker run --rm -it \   
   -p 8000:8000 \
   -v $HOME/.cache/huggingface:/huggingface \
   -e HF_HOME=/huggingface \
   -e VLLM_CPU_KVCACHE_SPACE=4 \
   -e VLLM_CPU_OMP_THREADS_BIND=0-4 \
   -e MAX_MODEL_LENGTH=8192 \
   -e VLLM_LOGLEVEL=DEBUG \
   --entrypoint /bin/bash \
   vllm-${branch}:${tag}-${platform}
'
```

### Comments
The image does not run on M3 Mac. The following command fails to start the service and errs with error:  
```
libnuma: Warning: node argument -1 is out of range
get_mempolicy: Operation not permitted
```

