import torch
import torch.nn as nn
# different saving ways in practice

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(Self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = Model(n_input_features=6)
print(model.state_dict())# check how state_dict looks like:
# train the model
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())
# saving and loading a checkpoint
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optim_state": optimizer.state_dict()
}

#torch.save(checkpoint, ".\pytorchbasics\saveandload-Models\checkpoint.pth")
loaded_checkpoint = torch.load(".\pytorchbasics\saveandload-Models\checkpoint.pth")
epoch = loaded_checkpoint["epoch"]

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optim_state"])

print(optimizer.state_dict())

# for param in model.parameters():
#     print(param)
# # train your model... 

# FILE = ".\pytorchbasics\saveandload-Models\model.pth" # .pth is the common format to use meaning pytorch
# torch.save(model.state_dict(), FILE)

# # now we load our model from FILE
# # model = torch.load(FILE)
# # model.eval() # evaluation of model
# loaded_model = Model(n_input_features=6)
# loaded_model.load_state_dict(torch.load(FILE))
# loaded_model.eval()
# # using model to inspect a parameter
# for param in loaded_model.parameters():
#     print(param)