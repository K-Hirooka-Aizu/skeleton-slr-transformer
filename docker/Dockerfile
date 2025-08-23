# NVIDIA公式 CUDA + cuDNN ベース
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# タイムゾーン設定とパッケージ更新
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
        python3 python3-pip python3-venv \
        git curl wget tmux \
        openssh-server sudo && \
    mkdir /var/run/sshd && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# root パスワード設定（開発用: rootpass）
RUN echo 'root:rootpass' | chpasswd

# SSH 設定（rootログイン & パスワード認証許可）
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

# 作業ディレクトリ
WORKDIR /workspace

# Miniforge install
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
 && bash Miniforge3-$(uname)-$(uname -m).sh -b -p /root/conda \
 && rm Miniforge3-$(uname)-$(uname -m).sh

# conda のパスを通す
ENV PATH=/root/conda/bin:$PATH

# conda環境を作成
RUN conda create -n pt271slr python=3.10 -y

RUN echo ". /root/conda/etc/profile.d/conda.sh" >> /root/.bashrc \
 && echo "conda activate pt271slr" >> /root/.bashrc

# ポート公開（22=SSH）
EXPOSE 22

# デフォルトコマンド: sshd を起動
CMD ["/usr/sbin/sshd", "-D"]