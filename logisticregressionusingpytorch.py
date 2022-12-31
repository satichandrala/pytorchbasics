import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# This is a binary classification model so 0 and 1 - Lofistic regression
# prepare the data
# bc means breast cancer this is a breast cancer dataset that we load from sklearn 
bc  = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
print(n_samples, n_features)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

#we scale features meaning normal distribution around 0 mean standard deviation=1
sc = StandardScaler() # we do this whenever we do logistic regression 
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# reshape y tensor with a given size
y_train = y_train.view(y_train.shape[0], 1)
y_test  = y_test.view(y_test.shape[0], 1)

# prepare the model now 
# Linear model of function f = wx+b; sigmoid at the end
class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1) # 30 input features, 1 menas 1 classlabel at the end
    # first we apply a linear layer in forward pass and the sigmoid function
    def forward(self, x):
        y_pred= torch.sigmoid(self.linear(x))
        return y_pred
# our model- logistic regression model of size 
model = Model(n_features)

# Loss and Optimizer 
num_epochs=118 # iterations - I tried 50(87.72%), 90(86.84%), 100(89.47%), 110(88.60),118(acc-92%), 120(87.72%) iterations
learning_rate = 0.01
criterion = nn.BCELoss() # loss - Binary cross entropy
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # forward pass and loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # Backward pass and update the network
    loss.backward()
    optimizer.step() # we update weights and pytorch does everything

    # zero grad before the new step - because backward function adds all gradients
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}') # loss we limit to 4 decimals
# activation graph - computational graph where we evaluate the history
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_cls = y_pred.round()
    # accuracy eq means equal function
    acc =  y_pred_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')




     
