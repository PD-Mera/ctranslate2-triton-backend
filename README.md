# CTranslate2 Backend for Triton Inference Server (Support Whisper) 

In this project, I used docker Triton 22.12, for now, you can switch between translate models and whisper models by rename file to `ctranslate2.cc`

## Environments and Deployments

Pull and run docker with 

``` bash
docker pull nvcr.io/nvidia/tritonserver:22.12
docker run --gpus '"device=0,1"' -it --name dongtrinh.ctranslate -p8187:8000 -p8188:8001 -p8189:8002 -v/home/data2/dongtrinh/ctranslate2:/workspace/ --shm-size=16G nvcr.io/nvidia/tritonserver:22.12-py3
```

The scripts below will be executed in docker container

``` bash
## Install cmake
wget https://github.com/Kitware/CMake/releases/download/v3.26.0/cmake-3.26.0-linux-x86_64.tar.gz
tar -xzvf cmake-3.26.0-linux-x86_64.tar.gz
ln -sf $(pwd)/cmake-3.26.0-linux-x86_64/bin/* /usr/bin/

## Install intel-mkl
wget -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB> /dev/null | apt-key add -
sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
apt-get update
apt-get install -y intel-mkl-64bit-2020.0-088
rm -rf /var/lib/apt/lists/*

## Install Ctranslate2
git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd Ctranslate2
mkdir build && cd build
cmake .. -DWITH_CUDA=ON
make -j4
make install
cd ..

## Install pybind11
cd python
pip install -r install_requirements.txt
python3 setup.py bdist_wheel
pip install dist/*.whl
cd ..

## Install rapidjson
apt update
apt install rapidjson-dev

## Install Ctranslate2-triton-backend
git clone https://github.com/speechmatics/ctranslate2_triton_backend.git
cd ctranslate2_triton_backend

## Choose model type
# cp ctranslate2_src/ctranslate2_base.cc src/ctranslate2.cc 
cp ctranslate2_src/ctranslate2_whisper.cc src/ctranslate2.cc 
mkdir build && cd build
export BACKEND_INSTALL_DIR=$(pwd)/install
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRITON_ENABLE_GPU=1 -DCMAKE_INSTALL_PREFIX=$BACKEND_INSTALL_DIR
make install
cd ../..

## Setup backend
pip install ctranslate2 transformers

## Convert model with ctranslate2
mkdir -p /workspace/deploy_models/whisper/
ct2-transformers-converter --model openai/whisper-tiny --output_dir /workspace/deploy_models/whisper/model

## Deploy
export MODEL_DIR=/workspace/deploy_models/
export CUDA_VISIBLE_DEVICES=1 
export LD_PRELOAD=/opt/intel/compilers_and_libraries_2020.0.166/linux/compiler/lib/intel64_lin/libiomp5.so 
tritonserver --backend-directory $BACKEND_INSTALL_DIR/backends --model-repository $MODEL_DIR
```

## Inference

You can run `./ctranslate2_triton_backend/examples/whisper_client.py` to inference

``` bash
python ./ctranslate2_triton_backend/examples/whisper_client.py
```

# (OLD README) CTranslate2 Backend for Triton Inference Server

This is a [backend](https://github.com/triton-inference-server/backend) based on [CTranslate2](https://github.com/OpenNMT/CTranslate2) for NVIDIA's [Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server), which can be used to deploy translation and language models supported by CTranslate2 on Triton with both CPU and GPU capabilities.

It supports ragged and dynamic batching and setting of (a subset of) CTranslate decoding parameters in the model config.

## Building

Make sure to have [cmake](https://cmake.org) installed on your system.

1. Build and install CTranslate2: [https://opennmt.net/CTranslate2/installation.html#compile-the-c-library](https://opennmt.net/CTranslate2/installation.html#compile-the-c-library)
2. Build the backend 
```bash
mkdir build && cd build
export BACKEND_INSTALL_DIR=$(pwd)/install
cmake .. -DCMAKE_BUILD_TYPE=Release -DTRITON_ENABLE_GPU=1 -DCMAKE_INSTALL_PREFIX=$BACKEND_INSTALL_DIR
make install
```

This builds the backend into `$BACKEND_INSTALL_DIR/backends/ctranslate2`.

## Setting up the backend

First install the pip package to convert models: `pip install ctranslate2`. Then create a model repository, which consists of a configuration (config.pbtxt) and the converted model.  

For example for the [Helsinki-NLP/opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) HuggingFace transformer model, create a new directory e.g. `mkdir $MODEL_DIR/opus-mt-en-de`.
The model needs to be moved into a directory called `model` that is nested in a folder specifying a numerical version of the model:

```bash
ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir 1/model
```

The minimum configuration for the model and backend is the following, you can see an example configs in [examples/model_repo](examples/model_repo):

```protobuf
backend: "ctranslate2" # must be ctranslate2
name: "opus-mt-en-de" # must be the same as the model name
max_batch_size: 128 # can be optimised based on available GPU memory
input [
  {
    name: "INPUT_IDS"
    data_type: TYPE_INT32
    dims: [ -1 ]
    allow_ragged_batch: true # needed for dynamic batching
  }
]
output [
  {
    name: "OUTPUT_IDS"
    data_type: TYPE_INT32
    dims: [ -1 ]
  }
]

instance_group [{ kind: KIND_GPU, count: 1 }] # use KIND_CPU for CPU inference
dynamic_batching {
  max_queue_delay_microseconds: 5000 # can be tuned based on latency requirements
}
```

Start the tritonserver with the ctranslate backend and model repository: 
```bash
tritonserver --backend-directory $BACKEND_INSTALL_DIR/backends --model-repository $MODEL_DIR
```

The backend is set up to use ragged, [dynamic batching](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher).
This means you should send each input in it's own request and Triton will take care of batching, using the `dynamic_batching` and `max_batch_size` configuration to build appropriate batches, 
for best performance on GPUs. 

### Providing a target prefix

There are models that require a special prefix token for the decoder. For example the M2M models need a token that specifies the target language, or sometimes it might be useful to start the translation with a specific prefix. An example config can be found in the [facebook M2M config.pbtxt](examples/model_repo/facebook_m2m100_1.2B/config.pbtxt).


## Sending requests

The backend expects token IDs as input and output, which need to be INT32 or bigger data-types. This might include beginning-of-sentence and end-of-sentence token, depending on the model. 
No padding tokens need to be added though, the backend is taking care of padding and batching.

You can use the offical Triton clients to make requests, both HTTP and gRPC protocols are supported. We provide an example for a Python client [here](examples/client.py).

You can try the working translation by running
```bash
echo "How is your day going?" | python3 examples/client.py
```

In case you want to use a special prefix for decoding, the request also needs to have the input `TARGET_PREFIX` set, which could look like this in Python:

```python
inputs.append(
    tritonclient.grpc.InferInput("TARGET_PREFIX", prefix_ids.shape, np_to_triton_dtype(prefix_ids.dtype))
)
inputs[1].set_data_from_numpy(prefix_ids)
```

## Configuration

The backend exposes a few parameters for customisation, currently that is a subset of decoding options in the CTranslate2 C++ interface and some special ones that are useful for limiting inference compute requirements:

* `compute_type` overrides the numerical precision used for the majority of the Transformer computation, relates to [quantization types](https://opennmt.net/CTranslate2/quantization.html#quantization) in model conversion.
* `max_decoding_length_multiple` can be used to limit the number of output tokens as a multiple of input tokens. E.g if the longest input sequence is 10 tokens and `max_decoding_length_multiple: "2"` decoding is limitted to 20 tokens.

Decoding parameters passed through to CTranslate, more to be added:

* `beam_size`
* `repetition_penalty`

The parameters can be set like this in the config.pbtxt:

```protobuf
parameters [
  {
    key: "compute_type"
    value {
      string_value: "float16" # optional, can be used to force a compute type
    }
  },
  {
    key: "max_decoding_length_multiple"
    value {
      string_value: "2" 
    }
  },
    {
    key: "beam_size"
    value {
      string_value: "4" 
    }
  },
    {
    key: "repetition_penalty"
    value {
      string_value: "1.5"
    }
  }
]
```
