# NVIDIA公式 CUDA + cuDNN ベース
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# タイムゾーン設定とパッケージ更新
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i -e 's/^APT/# APT/' -e 's/^DPkg/# DPkg/' \ 
    /etc/apt/apt.conf.d/docker-clean \
    && apt-get update && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
        python3 \
        curl \
        wget \
        tmux \
        sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /workspace

# Miniforge install
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
 && bash Miniforge3-$(uname)-$(uname -m).sh -b -p /root/conda \
 && rm Miniforge3-$(uname)-$(uname -m).sh

# conda のパスを通す
ENV PATH=/root/conda/bin:$PATH
ENV TERM=xterm-256color

# conda環境を作成
RUN conda create -n pt271slr python=3.10 -y

RUN echo ". /root/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
 && echo "conda activate pt271slr" >> /root/.bashrc

CMD ["tail", "-f", "/dev/null"]