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

for param in model.parameters():
    print(param)
# train your model... 

FILE = ".\pytorchbasics\saveandload-Models\model.pth" # .pth is the common format to use meaning pytorch
torch.save(model.state_dict(), FILE)

# now we load our model from FILE
# model = torch.load(FILE)
# model.eval() # evaluation of model
loaded_model = Model(n_input_features=6)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()
# using model to inspect a parameter
for param in loaded_model.parameters():
    print(param)