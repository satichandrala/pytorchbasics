import torch

# create tensor

x = torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0, requires_grad=True)

# FORWARD PASS

y_hat = w*x
loss = (y_hat - y)**2

print(loss)

# backward pass - pytorch does this automaticallyY

loss.backward()
print(w.grad)

## update weights
# next forward and backward