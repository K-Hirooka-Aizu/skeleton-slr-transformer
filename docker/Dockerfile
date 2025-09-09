# NVIDIA公式 CUDA + cuDNN ベース
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# タイムゾーン設定とパッケージ更新
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i -e 's/^APT/# APT/' -e 's/^DPkg/# DPkg/' \ 
    /etc/apt/apt.conf.d/docker-clean \
    && apt update && apt upgrade -y \
    && apt install --no-install-recommends -y \
        python3.12-dev \
        curl \
        wget \
        sudo \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /workspace

# Miniforge install
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/download/25.3.1-0/Miniforge3-$(uname)-$(uname -m).sh" \
 && bash Miniforge3-$(uname)-$(uname -m).sh -b -p /root/conda \
 && rm Miniforge3-$(uname)-$(uname -m).sh

# conda のパスを通す
ENV PATH=/root/conda/bin:$PATH

# conda環境を作成
COPY requirements.txt .
RUN conda create -n pt271slr python=3.10 -y \
 && echo "source /root/conda/bin/activate pt271slr" >> /root/.bashrc \
 && /bin/bash -c "source /root/conda/bin/activate pt271slr && pip install -r requirements.txt"

RUN echo ". /root/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
 && echo "conda activate pt271slr" >> /root/.bashrc

CMD ["tail", "-f", "/dev/null"]