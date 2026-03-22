import torch
import numpy as np


# tensor 생성
data = [[1,2],[3,4]]
x_data = torch.tensor(data)

# numpy array를 tensor로 변환
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 다른 tensor로 부터 생성하기 _like : 크기가 같은 값의 텐서 생성
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float)
print(f"Random Tensor:\n{x_rand}\n")

shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor : \n {rand_tensor}\n")
print(f"Random Tensor : \n {ones_tensor}\n")
print(f"Zeros Tensor : \n {zeros_tensor}\n")


# Tensor의 속성 dtype은 tensor안에 들어있는 요소의 타입
tensor = torch.rand(3,4)

print(f"Shape of tensor {tensor.shape}")
print(f"Dtype of tensor {tensor.dtype}")
print(f"Devices of tensor {tensor.device}")


# Tensor 연산
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor= torch.ones(4,4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")
print(f"Last row: {tensor[:,-1]}")
tensor[:,1] = 0
print(tensor)

# torch.cat과 torch.concat은 동일
t1 = torch.cat([tensor, tensor,tensor], dim=1)
print(t1)

# 산술 연산
# 두 텐서 간의 행렬 곱(matrix multiplication)을 계산합니다. y1, y2, y3은 모두 같은 값을 갖습니다.
# ``tensor.T`` 는 텐서의 전치(transpose)를 반환합니다.
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T,out=y3)

# 요소별 곱
z1=tensor*tensor
z2=tensor.mul(tensor)
z3 = torch.rand_like(z1)
torch.mul(z1,z1,out=z3)

# 단일-요소(single-element) 텐서 텐서의 모든 값을 하나로 집계(aggregate)하여 요소가 하나인 텐서의 경우, 
# item() 을 사용하여 Python 숫자 값으로 변환할 수 있습니다: item은 값이 하나일떄만 쓸 수 있다.
agg = tensor.sum()
agg_item=agg.item()
print(agg_item, type(agg_item))

# 바꿔치기 연산
print(f"{tensor}\n")
tensor.add_(5)
print(tensor)

# Numpy 변환
# 텐서를 넘파이 배열로 변환
t = torch.ones(5)
n = t.numpy()

# 텐서의 변경 사항이 NumPy 배열에 반영됩니다.
t.add_(1)

# 넘파이를 텐서로
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n,1,out=n)

