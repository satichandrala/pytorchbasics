import torch
import numpy as np

a = torch.ones(5)
print(a)
b = a.numpy()  # be careful in a cpu because it'll change the value completely pointing to the same memory location
print(b)

a.add_(1)
print(a)
print(b)

# OTHER WAY FROM numpy to tensor

c = np.ones(5)
print(c)
d = torch.from_numpy(c)
print(d)

c += 1
print(c)
print(d)
# Intiliazing a cuda device if available
# Points to note # NUMPY CAN only HANDLE CPU TENSORS AND yOU CANNOT HANDLE gpu TENSORS
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y =  torch.ones(5)
    y = y.to(device)
    z = x + y # WILL BE PERFORMED ON gpU and is much faster 
    z = z.to("cpu") # now transferred the value to cpu 

