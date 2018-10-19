import numpy as np
import torch

x = torch.ones(2, 2, requires_grad = True)
print(x)

y = x + 2
print(y)

y = y*y*3
out = y.mean()
print(y, out)

out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x*2
while y.data.norm() < 1000:
   y = y*2

print(y)