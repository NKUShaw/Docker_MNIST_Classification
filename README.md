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

## 📈 Output

```bash
Epoch 0, Batch 0, Loss 2.2971
Epoch 0, Batch 100, Loss 0.2794
Epoch 0, Batch 200, Loss 0.1680
Epoch 0, Batch 300, Loss 0.2421
Epoch 0, Batch 400, Loss 0.1238
Epoch 0, Batch 500, Loss 0.1077
Epoch 0, Batch 600, Loss 0.1662
Epoch 0, Batch 700, Loss 0.0913
Epoch 0, Batch 800, Loss 0.0764
Epoch 0, Batch 900, Loss 0.0495
...
Test Accuracy: 97.48%
```
