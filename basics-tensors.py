import torch

x =  torch.rand(2, 2) # one dimensional tensor
y = torch.rand(2, 2) #like a 2d matrix a 2dimensional tensor
z = torch.empty(2, 2, 3) #3d tensor and you can create an empty tensor with more dimensions



# printing a random tensor using rand
u = torch.rand(2, 2, dtype=torch.float16)
v = torch.rand(2,2)
print(u, v)

w = torch.tensor([2.5, 0.1]) # MAKING A TENSOR
# Addition
a = u + v
#Basic Operations with Tensors
b =  torch.add(u, v) # Using torch to make arithmetic operations 
c = torch.mul(u,v)

u.add_(x)


print(b)

# Slicing of tensors

d = torch.rand(5, 3)
print(d)
print(d[:, 0]) # PRINT ALL ROWS BUT ONLY COLUMN 0
print(d[1, :]) # PRINT SECOND ROW AND ALL THE COLUMNS IN "2ND ROW"

print(d[1, 1]) # print the specific element of the tensor

print(d[1,1].item())

# reshaping the tensor

e = torch.rand(4,4)
print(e)
# RESHAPING USING VIEW METHOD
#f = e.view(16) #giving a size 16 reshaping the tensor as a one row of all 16 elements
# if we don't want to put the 

f = e.view(-1, 8) # DOUBT ASK timo how's -1, 8 a two dimensional one
print(f.size())

# for optimization - if variable needs to be optimized, then use requires_grad=True
x = torch.ones(5, requires_grad=True)
print(x)

