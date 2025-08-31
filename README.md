# Template Repository for AI

# BUILD container by Docker
```bash
docker build -t dl-gpu ./docker/
docker run --gpus all -d -v $(pwd):/workspace --name my-dl dl-gpu
docker exec -it my-dl bash
```

# BUILD container by nerdctl
```bash
nerdctl build -t dl-gpu ./docker/
nerdctl run --gpus all -d -v $(pwd):/workspace --name my-dl dl-gpu
nerdctl exec -it my-dl bash
```

# BUILD nerdctl
```bash
wget https://github.com/containerd/nerdctl/releases/download/v2.1.3/nerdctl-full-2.1.3-linux-amd64.tar.gz -O ~/nerdctl-full-2.1.3-linux-amd64.tar.gz \
&& mkdir -p ~/.local && tar Cxzvvf ~/.local ~/nerdctl-full-2.1.3-linux-amd64.tar.gz && rm -f ~/nerdctl-full-2.1.3-linux-amd64.tar.gz

export PATH=$HOME/.local/bin:$PATH

containerd-rootless-setuptool.sh install

CONTAINERD_NAMESPACE=default containerd-rootless-setuptool.sh install-buildkit-containerd
```
