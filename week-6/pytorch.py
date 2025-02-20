import torch

print('\n============ Task 3-1 ============\n')

tensor = torch.tensor([[2, 3, 1], [5, -2, 1]])

print(tensor.shape)
print(tensor.dtype)


print('\n============ Task 3-2 ============\n')

tensor = torch.rand(3, 4, 2)

print(tensor.shape)
print(tensor)


print('\n============ Task 3-3 ============\n')

tensor = torch.ones(2, 1, 5)

print(tensor.shape)
print(tensor)


print('\n============ Task 3-4 ============\n')

tensorA = torch.tensor([[1, 2, 4], [2, 1, 3]])
tensorB = torch.tensor([[5], [2], [1]])

print(torch.matmul(tensorA, tensorB))


print('\n============ Task 3-5 ============\n')

tensorA = torch.tensor([[1, 2], [2, 3], [-1, 3]])
tensorB = torch.tensor([[5, 4], [2, 1], [1, -5]])

print(tensorA.mul(tensorB))