import torch
import torch.nn as nn
import numpy as np

# create a function for softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
output = softmax(x)
print('softmax numpy:', output)
# softmax numpy: [0.65900114 0.24243297 0.09856589]

# using pytorch

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0)
print(outputs)

