# MNIST CNN with PyTorch in Docker

This project trains and evaluates a CNN on the MNIST dataset using PyTorch, all inside Docker.

## 🔧 Requirements

- [Docker](https://www.docker.com/products/docker-desktop) installed

## 🚀 Run

1. Clone the repository:

```bash
git clone https://github.com/NKUShaw/Docker_MNIST_Classification.git
cd Docker_MNIST_Classification
```

2. Build the Docker image:

```bash
docker build -t mnist-pytorch .
```

3. Run the container:

```bash
docker run --gpus all --rm mnist-pytorch
```
