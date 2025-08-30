# Template Repository for AI

# BUILD Docker
```bash
docker build -t dl-gpu-ssh ./docker/
docker run --gpus all -d -v $(pwd):/workspace --name my-dl dl-gpu-ssh
docker update --restart=always my-dl
docker exec -it my-dl bash
```

# BUILD nerdctl
```bash
nerdctl build -t dl-gpu-ssh ./docker/
nerdctl run --gpus all -d -v $(pwd):/workspace --name my-dl dl-gpu-ssh
nerdctl update --restart=always my-dl
nerdctl exec -it my-dl bash
```
