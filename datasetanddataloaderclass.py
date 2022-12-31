# training loop
#   better way to work on a large dataset is to divide them into batches
#   epoch = 1 forward and backward pass of al training samples
#   batch_size = number of training samples in one forward and backward pass
#   number of iterations = number of passes, each pass using [batch_size] number of samples
#   e.g., 100 sample, batch_size=20 --> 5 iterations for 1 epoch

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math

# Datasets
class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./pytorchbasics/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        print(xy)
        self.x = torch.from_numpy(xy[:, 1:]) #except first column, when you write torch.from_numpy we are converting to tensor
        self.y = torch.from_numpy(xy[:, [0]]) # size n_samples, 1
        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

        #dataset[0]
    def __len__(self):
        return self.n_samples
        #len(dataset)
dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data # unpack the data into features and labels
# print(features, labels) # prints 1 row vector and a tensor column 

# Dataloader we use the builtin DataLoader class
# if __name__ == '__main__':  
dataloader = DataLoader(dataset= dataset, batch_size=4, shuffle=True, num_workers=2) # shuffle will shuffle will be good for training, num_workers uses multiple subprocesses and loading will be easier
#     datatiter = iter(dataloader)
#     data = next(datatiter) 
# # unpack this
#     features, labels= data
#     print(features, labels) # to see if the above is working , in output you'll see feature vectors and labels tensors
# training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
                if __name__ == '__main__':
                    for i, (inputs, labels) in enumerate(dataloader): # enumerate gives us the index and also the inputs and the labels
        # forward and backward pass and update
                        if (i +1) % 5 == 0: # every 5th step you need to print some info
                             print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
torchvision.datasets.MNIST()
#fashion MNIST
# CIFAR dataset
# coco dataset
