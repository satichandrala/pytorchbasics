# MNIST, DataLoader, 
# Transformation, 
# Multilayer Neural net; 
# aCTIVATION FUCNTION
# loss and optimizer 
# Training Loop (batch Training)
# model evaluation
# GPU Support
# Using tensorboard: 
# import tensorboard from utils and Summary writer give directory and when displaying images exit using sys and plot the data.
# 

# import the libraries needed
import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/mnist2") # updated learning rate to 0.01 and so renamed the folder to see different results on this
# device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28*28 784 because our images have the size 28*28
hidden_size = 100 # we can try different sizes  
num_classes = 10 # we have 10 different classes of digits 0 to 9
num_epochs = 2 # so that training doesn't take too long
batch_size = 100
learning_rate = 0.01

# MNIST - can have from the pytorch training library
train_dataset = torchvision. datasets.MNIST(root = './data', train=True,
transform=transforms.ToTensor(), download=True)

test_dataset = torchvision. datasets.MNIST(root = './data', train=False,
transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # shuffle=True is good for training

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # shuffle doesn't matter for test datasets

# we take a batch of the data
examples = iter(test_loader)
# samples, labels = next(examples)
# print(samples.shape, labels.shape)
example_data, example_targets = next(examples)


# output
#  torch.Size([100, 1, 28, 28]) torch.Size([100])
#  in the output 100 is size of samples in batch , 1 means 1 channel because MNIST has no colored channels, 28*28 is image sizes
# torch.Size ([100]) for each class values we have one label
# plotting the examples data with matpllotlib
for i in range(6):
     plt.subplot(2, 3, i+1) # 2 - rows 3- columns - index - i+1
     plt.imshow(example_data[i][0], cmap='gray') # showing actual data samples[i], [0] is the 1st channel, cmap is columnmap
#plt.show()
img_grid = torchvision.utils.make_grid(example_data)
writer.add_image('mnist_images', img_grid)
writer.close() # this will make sure all the outputs are being flushed here
#sys.exit()

# now we classify thes - we setup a fully connected neural network with one hidden layer

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):  # num_classes is output size
        super(NeuralNet, self).__init__() #we call super init
        # we create our layers now
        self.l1 = nn.Linear(input_size, hidden_size)
        # after the first layer we use an activation fucntion
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes) # another layer - now input size is hidden size and output size is number of classes

    def forward(self, x):
        out = self.l1(x) # here l1 with sample x
        out = self.relu(out)
        out = self.l2(out)
        # here at the end we don't need an activation problem
        # Cross Entropy loss will apply softmax we don't need to apply softmax separately
        return out
model = NeuralNet(input_size, hidden_size, num_classes)

# create the loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# making a graph to display on tensorboard

writer.add_graph(model, example_data.reshape(-1, 28*28)) # we give model and the input batch data with reshaping data 
writer.close()
#sys.exit()

# training Loop
n_total_steps = len(train_loader)
running_loss = 0.0
running_correct = 0
for epoch in range(num_epochs):
    # now we will loop over all bacthes
    for i, (images, labels) in enumerate(train_loader):
        # now we reshape the images because we have 100 of 1 channel 28*28 images
        # first number of batches=100 and then input size is 784 
        images = images.reshape(-1, 28*28).to(device) # -1 is the first dimension then tensor will find it automatically for us
        labels = labels.to(device)

        # forward
        outputs = model(images)
        # we calculate the loss
        loss = criterion(outputs, labels) # predicted putput and actual labels

        # backward loop

        optimizer.zero_grad() # to empty the values in the gradient attribute
        loss.backward() # backpropogation 
        optimizer.step() # update step, updates the parameters

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        running_correct += (predicted == labels).sum().item()
# print the loss - every 100th step we print some info
        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss/100, epoch*n_total_steps + i)
            writer.add_scalar('accuracy', running_correct/100, epoch*n_total_steps + i)
            running_loss = 0.0
            running_correct = 0
# for precision recall curve
labels = []
preds = []
# we will show mean loss in tensor board
            # now done with the training loop
# we make a test loop and evaluation 
# we don'T want to compute  the gradients for all the steps we do s
# so we wrap with a no grad

with torch.no_grad():
    n_correct   = 0 # initialization
    n_samples   = 0
    for images, labels1 in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels1 = labels1.to(device)
        outputs = model(images)

# returns the value, index
        _, predictions = torch.max(outputs.data, 1) # dimension is 1, _ is class label no not needed any value b/c we're interested in index
        n_samples += labels1.size(0) # number of samples in current batch
        n_correct += (predicted == labels1).sum().item() # += adding one each time
        # for batch evaluation
        class_predictions = [F.softmax(output, dim=0) for output in outputs]
        preds.append(class_predictions)
        labels.append(predicted)
# we calculate softmax explicitly for outputs
    preds = torch.cat([torch.stack(batch) for batch in preds])
    labels = torch.cat(labels) # concatenate all the labels into a 1d tensor


    acc = 100.0 * n_correct/ n_samples
    print(f'accuracy = {acc}')

    classes = range(10)
    for i in classes:
        labels_i = labels == i
        preds_i = preds[:, i]
        writer.add_pr_curve(str(i), labels_i, preds_i, global_step=0)
        writer.close()
