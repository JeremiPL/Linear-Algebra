import torch

x = torch.tensor(0.0, requires_grad = True)
y = torch.tensor(0.0, requires_grad = True)
z = torch.tensor(2.0, requires_grad = True)
f = (5*x + 3*x**3*z**2*y**3 - 5*z - 4*x**2*y**3*z) / (5*y*x**2*z**2 + 3*y**2 + 3*y + 1)
f.backward()
print(x.grad)
print(y.grad)
print(z.grad)
