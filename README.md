# Template Repository for AI

# BUILD Docker
```bash
docker build -t dl-gpu-ssh ./docker/
docker run --gpus all -d -p 2222:22 -v $(pwd):/workspace --name my-dl dl-gpu-ssh
```
