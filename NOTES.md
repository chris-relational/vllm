# TL; DR
I log here my reading conprehension notes and tasks regarding the repository.  
The `var` directory is contained in `.gitignore` so it does not mix with the repo code.



# `docker/Dockerfile.arm`
The command `ENV LD_PRELOAD="/usr/lib/aarch64-linux-gnu/libtcmalloc_minimal.so.4"`
sets the container environment variable LD_PRELOAD to the reference shared library 
(note this is an Ubuntu image created on an arm64 host).  

__(ChatGPT):__ LD_PRELOAD instructs the dynamic linker to load a shared library before any other library when running executables.  
It allows you to override functions in system libraries or inject extra functionality without changing the application binary.  

`tcmalloc_minimal` is the minimal version of Google’s TCMalloc library (Thread-Caching Malloc), 
an optimized memory allocator from the "Google Performance Tools suite". It provides faster malloc/free 
than the default system allocator (glibc malloc). Helps improve performance of memory-intensive apps.

The vLLM documentation on docker installation, contains a build command for x86 cpus:
```bash
$ docker build -f docker/Dockerfile.cpu --tag vllm-cpu-env --target vllm-openai .
```

If we replace (as requested) `Dockerfile.cpu` by `Dockerfile.arm` the build fails. This is because `--target vllm-openai`
refers to a stage in `Dockerfile.cpu` that is not contained in `Dockerfile.arm`. 
See the [docker documentation](https://docs.docker.com/build/building/multi-stage/) on multi-staged builds (there
it is also explained what happens with dockerfiles with multiple `FROM` commands).  

Here we use the command `$ docker build -f docker/Dockerfile.cpu --tag vllm-openai:arm .`. 



# Additional installation requirements
1. For the chat REPL of `transformers` you need to `pip install accelerate`. This is a framework for distributing inference to
multiple nodes

2. To run the `generate` SDK it is recommended to `pip install bitsandbytes` (by Huggingface).  
   `bitsandbytes` has methods for quantizing (when loading to memory) LLMs that greately improves performance.  




