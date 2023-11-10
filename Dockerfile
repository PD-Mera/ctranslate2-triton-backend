# change tritonserver version accordingly
FROM nvcr.io/nvidia/tritonserver:23.08-py3

# Install libraries required for compilation
RUN mkdir installations
WORKDIR /opt/tritonserver/installations
# CMake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.0/cmake-3.26.0-linux-x86_64.tar.gz
RUN tar -xzvf cmake-3.26.0-linux-x86_64.tar.gz
RUN ln -sf $(pwd)/cmake-3.26.0-linux-x86_64/bin/* /usr/bin/
RUN rm -f cmake-3.26.0-linux-x86_64.tar.gz

# MKL - required for build even if we intend to build just for GPU
RUN wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/adb8a02c-4ee7-4882-97d6-a524150da358/l_onemkl_p_2023.2.0.49497_offline.sh
RUN chmod 755 l_onemkl_p_2023.2.0.49497_offline.sh
RUN ./l_onemkl_p_2023.2.0.49497_offline.sh -a -s --eula accept
RUN rm -f l_onemkl_p_2023.2.0.49497_offline.sh

# RapidJSON - TODO: find binary 
RUN apt update
RUN apt install rapidjson-dev

# Python Libraries
RUN pip install --upgrade pip && pip install ctranslate2 transformers



WORKDIR /opt/tritonserver
COPY . ./ctranslate-triton-backend


# build CTranslate2 for Cuda compute
WORKDIR /opt/tritonserver/ctranslate-triton-backend/src/CTranslate2/build
RUN cmake .. -DWITH_CUDA=ON -DWITH_CUDNN=ON -DWITH_MKL=OFF
RUN make -j4
RUN make install

# install CTranslate Python requirements
WORKDIR /opt/tritonserver/ctranslate-triton-backend/src/CTranslate2/python
RUN pip install -r install_requirements.txt
RUN python3 setup.py bdist_wheel
RUN pip install dist/*.whl


# build ctranslate-triton-backend
WORKDIR /opt/tritonserver/ctranslate-triton-backend/src/ctranslate-triton-backend/build
RUN export BACKEND_INSTALL_DIR=$(pwd)/install
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DTRITON_ENABLE_GPU=1 -DCMAKE_INSTALL_PREFIX=$BACKEND_INSTALL_DIR
RUN make install
RUN mkdir -p /opt/tritonserver/backends/ctranslate2 && cp libtriton_ctranslate2.so /opt/tritonserver/backends/ctranslate2/libtriton_ctranslate2.so

# Everything is built now!
# EXAMPLE model conversion & deployment
WORKDIR /opt/tritonserver/ctranslate-triton-backend/examples/model_repo
RUN pip install torch sentencepiece
RUN ct2-transformers-converter --model Helsinki-NLP/opus-mt-en-de --output_dir Helsinki-NLP_opus-mt-en-de/1/model
RUN export CUDA_VISIBLE_DEVICES=0
RUN export export LD_PRELOAD=/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin/libiomp5.so
#RUN tritonserver --model-repository /opt/tritonserver/ctranslate-triton-backend/examples/model_repo &
