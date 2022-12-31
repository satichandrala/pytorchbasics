import torch
import torch.nn as nn

# Multiclass problem
class NeuralNet2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet2, self).__init__()
        self.linear1    =  nn.Linear(input_size, hidden_size)
        self.relu   = nn.ReLU()
        self.linear2    = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # no softmax since we are using torch
        return out

model = NeuralNet2(input_size=28*28, hidden_size=5, num_classes=3)
criterion   = nn.CrossEntropyLoss() # (this applies softmax)

# Binary Classification
class Neuralnet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1    =  nn.Linear(input_size, hidden_size)
        self.relu   = nn.ReLU()
        self.linear2    = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred

model = Neuralnet1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss()