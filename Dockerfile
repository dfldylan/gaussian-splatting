FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV TZ Asia/Shanghai
#ENV http_proxy=http://router4.ustb-ai3d.cn:3128
#ENV https_proxy=http://router4.ustb-ai3d.cn:3128
WORKDIR /root
RUN apt update && apt install -y wget git vim openssh-server net-tools
RUN wget -c https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p /${HOME}/miniconda3 && /${HOME}/miniconda3/bin/conda init
RUN conda env create --file environment.yml

EXPOSE 22 6009 8888