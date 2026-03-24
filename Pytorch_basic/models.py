import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

devices = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {devices} devices")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(devices)
# print(model)

X = torch.rand(1,28,28, device=devices)
logits = model(X)
# print(f"logit is {logits}")
predict = nn.Softmax(dim=1)(logits)
# print(f"predict is {predict}")
y_pred = predict.argmax(1)
# print(f"y_pred is{y_pred}")
print(y_pred.item())
