import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.nn.parameter import Parameter


tsr = torch.Tensor(3,5)
print(tsr)

import torch

a = torch.randn((3,3), requires_grad = True)

w1 = torch.randn((3,3), requires_grad = True)
w2 = torch.randn((3,3), requires_grad = True)
w3 = torch.randn((3,3), requires_grad = True)
w4 = torch.randn((3,3), requires_grad = True)

b = w1*a
c = w2*a

d = w3*b + w4*c

L = 10 - d

print("The grad fn for a is", a.grad_fn)
print("The grad fn for d is", d.grad_fn)

print(a.is_leaf, b.is_leaf, d.is_leaf, L.is_leaf)

# L = (10 -d).sum()
L = 10 -d

# L.backward()
L.backward(torch.ones(L.shape))
L.backward(torch.ones(L.shape))

print(a.grad)

m = nn.ReLU()
output = m(a)
