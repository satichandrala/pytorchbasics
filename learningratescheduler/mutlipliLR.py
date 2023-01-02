import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10,1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lambda1 = lambda epoch: 0.95
scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda1) # mutliply with the previous learning rate

print(optimizer.state_dict())
for epoch in range(5):
    optimizer.step()
    #validate...
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])