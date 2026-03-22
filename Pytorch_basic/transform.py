import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0,torch.tensor(y), value=1)) # 0차원에서 3번 위치에 value=1을 넣겠다. one-hot encoding _가 붙으면 원본텐서를 직접 바꾼다.
)