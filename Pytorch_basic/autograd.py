import torch

x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x,w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

print(z.requires_grad)
print(f"Gradient function for z ={z.grad_fn}")
print(f"Gradient function for loss ={loss.grad_fn}")
loss.backward()
print(w.grad)
print(b.grad)

with torch.no_grad():
    z = torch.matmul(x,w)+ b
print(z.requires_grad)

z = torch.matmul(x,w) + b
z_det = z.detach()