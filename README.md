# Transformer Based Sign Lnaguage Recognition

# BUILD container by Docker
```bash
docker build -t dl-gpu -f ./docker/Dockerfile .
docker run --gpus all -d -v $(pwd):/workspace --name my-dl dl-gpu
docker exec -it my-dl bash

docker stop my-dl && docker rm my-dl && docker rmi dl-gpu
```

# BUILD container by nerdctl
```bash
nerdctl build -t dl-gpu -f ./docker/Dockerfile .
nerdctl run --gpus all -d -v $(pwd):/workspace --name my-dl dl-gpu
nerdctl exec -it my-dl bash

nerdctl stop my-dl && nerdctl rm my-dl && nerdctl rmi dl-gpu
```

# BUILD nerdctl
```bash
wget https://github.com/containerd/nerdctl/releases/download/v2.1.3/nerdctl-full-2.1.3-linux-amd64.tar.gz -O ~/nerdctl-full-2.1.3-linux-amd64.tar.gz \
&& mkdir -p ~/.local && tar Cxzvvf ~/.local ~/nerdctl-full-2.1.3-linux-amd64.tar.gz && rm -f ~/nerdctl-full-2.1.3-linux-amd64.tar.gz

export PATH=$HOME/.local/bin:$PATH

containerd-rootless-setuptool.sh install

CONTAINERD_NAMESPACE=default containerd-rootless-setuptool.sh install-buildkit-containerd
```
