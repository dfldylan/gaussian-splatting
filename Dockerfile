FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL maintainer="dfldylan@qq.com"
ENV TZ Asia/Shanghai
#ENV http_proxy=http://router4.ustb-ai3d.cn:3128
#ENV https_proxy=http://router4.ustb-ai3d.cn:3128
RUN apt-get update && apt-get install -y wget git vim openssh-server net-tools libgl-dev libglm-dev freeglut3-dev
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create -y -n gaussfluids python=3.7.13
RUN echo "source /miniconda/bin/activate gaussfluids" > ~/.bashrc
ENV PATH /opt/conda/envs/gaussfluids/bin:$PATH
RUN conda install -n gaussfluids -c pytorch -c conda-forge -c defaults \
    cudatoolkit=11.6 \
    plyfile=0.8.1 \
    pip=22.3.1 \
    pytorch=1.12.1 \
    torchaudio=0.12.1 \
    torchvision=0.13.1 \
    tensorboard=2.8 \
    tqdm \
    embree3
RUN conda run -n gaussfluids pip install nbconvert==7.4.0 jupyterlab open3d==0.16.0 matplotlib argparse_dataclass
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 22 6009 8888
VOLUME /workspace
CMD ["sh", "-c", "service ssh start && /miniconda/envs/gaussfluids/bin/jupyter lab --ip=0.0.0.0 --no-browser --allow-root --notebook-dir=/workspace"]
WORKDIR /workspace
