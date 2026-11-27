# [SSTAN]: Official Implementation of "Stack Transformer-Based Spatial-Temporal Attention Model for Dynamic Sign Language and Fingerspelling Recognition"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)


This repository contains the official PyTorch implementation for our paper:  
**Stack Transformer-Based Spatial-Temporal Attention Model for Dynamic Sign Language and Fingerspelling Recognition**  
Koki Hirooka, Abu Saleh Musa Miah, Tatsuya Murakami, Md. Al Mehedi Hasan, Yong Seok Hwang, Jungpil Shin  
*IEEE Open Journal of the Computer Society*, vol. 7, pp. 706–717, 2026. [[IEEE Xplore]](https://ieeexplore.ieee.org/document/11478319)

The MIT license covers the source code only. The datasets have their own terms of use; see [Datasets](#datasets).

## 📝 Abstract
Hand gesture-based Sign Language Recognition (SLR) serves as a crucial communication bridge between deaf and non-deaf individuals. While Graph Convolutional Networks (GCNs) are common, they are limited by their reliance on fixed skeletal graphs. To overcome this, we propose the Stacked Spatio-Temporal Attention Network (SSTAN), a novel Transformer-based architecture. Our model employs a hierarchical, stacked design that sequentially integrates Spatial Multi-Head Attention (MHA) to capture intra-frame joint relationships and Temporal MHA to model long-range inter-frame dependencies. This approach allows the model to efficiently learn complex spatio-temporal patterns without predefined graph structures. We validated our model through extensive experiments on diverse, large-scale datasets (WLASL, JSL, and KSL). A key finding is that our model, trained entirely from scratch, achieves state-of-the-art (SOTA) performance in the challenging fingerspelling categories (JSL and KSL). Furthermore, it establishes a new SOTA for skeleton-only methods on WLASL, outperforming several approaches that rely on complex self-supervised pre-training. These results demonstrate our model's high data efficiency and its effectiveness in capturing the intricate dynamics of sign language. The official implementation is available at our GitHub repository: https://github.com/K-Hirooka-Aizu/skeleton-slr-transformer

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

### Training and Evaluate the model.
Run the commands from the repository root after placing the data as described in [data/README.md](./data/README.md).

```bash
# wlasl100
python scripts/train.py data=wlasl100 model=postnorm_transformer epochs=1500 seed=42

# wlasl300
python scripts/train.py data=wlasl300 model=postnorm_transformer epochs=1500 seed=42

# wlasl1000
python scripts/train.py data=wlasl1000 model=postnorm_transformer epochs=1500 seed=42

# wlasl2000
python scripts/train.py data=wlasl2000 model=postnorm_transformer epochs=1500 seed=42

# Japanese / Korean fingerspelling
python scripts/train.py data=jsl0 model=postnorm_transformer epochs=1000 seed=42
python scripts/train.py data=ksl0 model=postnorm_transformer epochs=1000 seed=42
```

### Training and Evaluate the model with multi-run.
```bash
# wlasl100
python scripts/train.py -m data=wlasl100 model=postnorm_transformer epochs=1500 seed=0,1,2,3,4,5,6,7,8,9

# wlasl300
python scripts/train.py -m data=wlasl300 model=postnorm_transformer epochs=1500 seed=0,1,2,3,4,5,6,7,8,9

# wlasl1000
python scripts/train.py -m data=wlasl1000 model=postnorm_transformer epochs=1500 seed=0,1,2,3,4,5,6,7,8,9

# wlasl2000
python scripts/train.py -m data=wlasl2000 model=postnorm_transformer epochs=1500 seed=0,1,2,3,4,5,6,7,8,9
```

All model parameters and dataset configurations are managed via **Hydra YAML files**.

The full configuration hierarchy is available under the `scripts/conf/` directory.
This setup enables reproducible experiments and flexible parameter control by editing configuration files or overriding them via the command line.

## Datasets

| Dataset | Language | Classes | Signers | Keypoints | Source |
|---|---|---|---|---|---|
| WLASL | ASL (word level) | 100 / 300 / 1000 / 2000 | 119 | OpenPose, 55 joints | [Official WLASL](https://dxli94.github.io/WLASL/) |
| JSLv2 | Japanese fingerspelling | 76 | 15 | MediaPipe hand, 21 joints | Our laboratory |
| KSL0 | Korean fingerspelling | 31 | 20 | MediaPipe hand, 21 joints | Our laboratory ([Shin et al., 2024](https://ieeexplore.ieee.org/document/10529244)) |

Download links, the directory layout, and the file format are described in [data/README.md](./data/README.md).

### Terms of use of the data
- **WLASL**: follow the license and terms of the [official WLASL repository](https://github.com/dxli94/WLASL).
- **JSLv2 / KSL0**: the released files contain only anonymized skeleton coordinates (no videos, images, or names). The formal terms of use are currently being confirmed with the co-authors. Until they are finalized, please contact us via GitHub Issues before using the data, and cite the papers listed in [Citation](#citation).
- **Note on the fingerspelling data**: the signers of JSLv2 and KSL0 are non-native signers. The KSL0 signers are members of our laboratory who performed the characters while looking at reference images; the JSLv2 signers performed them after watching a model demonstration. Their hand shapes may differ from those of native signers.

## Citation

If you use this code or the JSLv2 data, please cite:

```bibtex
@article{hirooka2026sstan,
  author  = {Hirooka, Koki and Miah, Abu Saleh Musa and Murakami, Tatsuya and Hasan, {Md. Al Mehedi} and Hwang, Yong Seok and Shin, Jungpil},
  title   = {Stack Transformer-Based Spatial-Temporal Attention Model for Dynamic Sign Language and Fingerspelling Recognition},
  journal = {IEEE Open Journal of the Computer Society},
  year    = {2026},
  volume  = {7},
  pages   = {706--717},
  doi     = {10.1109/OJCS.2026.3682330}
}
```

If you use the KSL0 data, please also cite:

```bibtex
@article{shin2024korean_ksl0,
  author  = {Shin, Jungpil and Miah, Abu Saleh Musa and Akiba, Yuto and Hirooka, Koki and Hassan, Najmul and Hwang, Yong Seok},
  title   = {Korean Sign Language Alphabet Recognition Through the Integration of Handcrafted and Deep Learning-Based Two-Stream Feature Extraction Approach},
  journal = {IEEE Access},
  year    = {2024},
  volume  = {12},
  pages   = {68303--68318},
  doi     = {10.1109/ACCESS.2024.3399839}
}
```

## Feedback

If you encounter any issues or unexpected behavior, please report them via GitHub Issues.

To help us reproduce the problem, please include:

* environment details (OS, GPU, CUDA, library versions)
* the exact command/configuration used
* error logs or stack traces

We appreciate your feedback and will do our best to address reported issues.

