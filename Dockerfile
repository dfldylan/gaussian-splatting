FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
LABEL maintainer="dfldylan@qq.com"
ENV TZ Asia/Shanghai
#ENV http_proxy=http://router4.ustb-ai3d.cn:3128
#ENV https_proxy=http://router4.ustb-ai3d.cn:3128
RUN echo 'Acquire::Retries "10";' > /etc/apt/apt.conf.d/80retries
RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirrors.mit.edu/ubuntu|g' /etc/apt/sources.list \
    && sed -i 's|http://security.ubuntu.com/ubuntu|http://mirrors.mit.edu/ubuntu|g' /etc/apt/sources.list
RUN apt-get update && apt-get install -y wget git vim openssh-server net-tools libgl-dev
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh \
    && bash /miniconda.sh -b -p /miniconda \
    && rm /miniconda.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda create -n gaussian_splatting python=3.7.13
RUN echo "source activate gaussian_splatting" > ~/.bashrc
ENV PATH /opt/conda/envs/gaussian_splatting/bin:$PATH
RUN conda install -n gaussian_splatting -c pytorch -c conda-forge -c defaults \
    cudatoolkit=11.6 \
    plyfile=0.8.1 \
    pip=22.3.1 \
    pytorch=1.12.1 \
    torchaudio=0.12.1 \
    torchvision=0.13.1 \
    tensorboard=2.8 \
    tqdm
RUN conda run -n gaussian_splatting pip install nbconvert==7.4.0 jupyterlab open3d==0.16.0 matplotlib
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 22 6009 8888
VOLUME /workspace
