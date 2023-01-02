import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1
model = nn.Linear(10,1)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) # decays the learning rate each epoch by gamma

print(optimizer.state_dict())
for epoch in range(5):
    optimizer.step()
    #validate...
    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])