# [SSTAN]: Official Implementation of "Stack Transformer-Based Spatial-Temporal Attention Model for Dynamic Sign Language and Fingerspelling Recognition"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** The code and pre-trained models will be made publicly available upon the acceptance of our paper.

This repository contains the official PyTorch implementation for our paper:  
**Stack Transformer Based Spatial-Temporal Attention Model for Dynamic Sign Language and Fingerspelling Recognition** Koki HIROOKA, Abu Saleh Musa Miah, Tatsuya Murakami, Yong Seok Hwang, Jungpil Shin  

[IEEE OJ-CS] (https://ieeexplore.ieee.org/document/11478319)

## 📝 Abstract
Hand gesture-based Sign Language Recognition (SLR) serves as a crucial communication bridge between deaf and non-deaf individuals. While Graph Convolutional Networks (GCNs) are common, they are limited by their reliance on fixed skeletal graphs. To overcome this, we propose the Stacked Spatio-Temporal Attention Network (SSTAN), a novel Transformer-based architecture. Our model employs a hierarchical, stacked design that sequentially integrates Spatial Multi-Head Attention (MHA) to capture intra-frame joint relationships and Temporal MHA to model long-range inter-frame dependencies. This approach allows the model to efficiently learn complex spatio-temporal patterns without predefined graph structures. We validated our model through extensive experiments on diverse, large-scale datasets (WLASL, JSL, and KSL). A key finding is that our model, trained entirely from scratch, achieves state-of-the-art (SOTA) performance in the challenging fingerspelling categories (JSL and KSL). Furthermore, it establishes a new SOTA for skeleton-only methods on WLASL, outperforming several approaches that rely on complex self-supervised pre-training. These results demonstrate our model's high data efficiency and its effectiveness in capturing the intricate dynamics of sign language. The official implementation is available at our GitHub repository: t tps://github.com/K-Hirooka-Aizu/skeleton-slr-transformer

## How to run
### (optional) Build Container
#### Build container by Docker
```bash
docker build -t dl-gpu -f ./docker/Dockerfile .
docker run --gpus all -d -v $(pwd):/workspace --name my-dl dl-gpu
docker exec -it my-dl bash

docker stop my-dl && docker rm my-dl && docker rmi dl-gpu
```

#### Build container by nerdctl
```bash
nerdctl build -t dl-gpu -f ./docker/Dockerfile .
nerdctl run --ipc host --gpus all -d -v $(pwd):/workspace --name my-dl dl-gpu
nerdctl exec -it my-dl bash

nerdctl stop my-dl && nerdctl rm my-dl && nerdctl rmi dl-gpu
```

#### Build nerdctl
```bash
wget https://github.com/containerd/nerdctl/releases/download/v2.1.3/nerdctl-full-2.1.3-linux-amd64.tar.gz -O ~/nerdctl-full-2.1.3-linux-amd64.tar.gz \
&& mkdir -p ~/.local && tar Cxzvvf ~/.local ~/nerdctl-full-2.1.3-linux-amd64.tar.gz && rm -f ~/nerdctl-full-2.1.3-linux-amd64.tar.gz

export PATH=$HOME/.local/bin:$PATH

containerd-rootless-setuptool.sh install

CONTAINERD_NAMESPACE=default containerd-rootless-setuptool.sh install-buildkit-containerd
```

### Training the model
```bash
# wlasl100
python script/train.py data=wlasl100 model=postnorm_transformer epochs=1500 seed=42
# wlasl300
python script/train.py data=wlasl300 model=postnorm_transformer epochs=1500 seed=42
# wlasl1000
python script/train.py data=wlasl1000 model=postnorm_transformer epochs=1500 seed=42
# wlasl2000
python script/train.py data=wlasl2000 model=postnorm_transformer epochs=1500 seed=42
```