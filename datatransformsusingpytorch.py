# convert images or numpy arrays to tensors

'''
Transforms can be applied to PIL images, tensors, ndarrays, or custom data
during creation of the DataSet
complete list of built-in transforms: 
https://pytorch.org/docs/stable/torchvision/transforms.html
On Images
---------
CenterCrop, Grayscale, Pad, RandomAffine
RandomCrop, RandomHorizontalFlip, RandomRotation
Resize, Scale
On Tensors
----------
LinearTransformation, Normalize, RandomErasing
Conversion
----------
ToPILImage: from tensor or ndrarray
ToTensor : from numpy.ndarray or PILImage
Generic
-------
Use Lambda 
Custom
------
Write own class
Compose multiple Transforms
---------------------------
composed = transforms.Compose([Rescale(256), # we cna compose multiple transforms which composes multiple transforms after each other
                               RandomCrop(224)])

'''

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import math

class WineDataset(Dataset):
    def __init__(self, transform = None):
        xy = np.loadtxt('./pytorchbasics/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        # print(xy)
        self.n_samples = xy.shape[0]

        # note we don't convert to Tensor here
        self.x = xy[:, 1:] #except first column, when you write torch.from_numpy we are converting to tensor
        self.y = xy[:, [0]] # size n_samples, 1

        self.transform =  transform

# for indexing the dataset

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
# if it's not None  we apply this
        if self.transform:
            sample = self.transform(sample) # all the change we need for our dataset
        return sample

        #dataset[0]
    def __len__(self):
        return self.n_samples
        #len(dataset)

# our Tensor class

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs; target

dataset = WineDataset(transform=None) # ToTensor converts the wine dataset to tensor using the ToTensor function and
#                                              # we see the output if we have transform = None, then it's still a numpy nd array
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels)) # output <class 'torch.Tensor'> <class 'torch.Tensor'>

#composed transform
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])

dataset = WineDataset(transform=composed)
first_data = dataset[0]
#features, labels = first_data #giving errors saying too many values to unpack expected 2
print(first_data)
print(type(first_data))
