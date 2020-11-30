# -*-coding:utf-8-*-
"""
https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


x = torch.ones(2, 2, requires_grad=True)
y = x + 2

print(x.grad)
print(type(x.grad))

v = torch.tensor([[0.1, 1.0], [0.1, 0.0001]], dtype=torch.float)
y.backward(v)

# print(y.grad_fn)
print(x.grad)
print(type(x.grad))

print(y.unsqueeze(0).shape)
print(y.shape)
