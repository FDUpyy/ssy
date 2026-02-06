import torch
import numpy as np
"""
a = torch.rand(2, 2, 3, 7)
print("原始输入张量:\n", a)

qurries = a[:, :, :, 2:]
keys_0 = values_0 = a[:, :, :, :a.shape[3]-2]
keys_1 = values_1 = a[:, :, :, 1:a.shape[3]-1]
keys_2 = values_2 = a[:, :, :, 2:]

keys = torch.cat((keys_0, keys_1, keys_2), dim=3)
print(keys.shape)
"""
delays = 3
for delay in range(delays):
    x = x + delay
    print(x)