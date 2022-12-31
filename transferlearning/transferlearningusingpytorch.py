# pretrained ResNet18 CNN, trained on more than a million images from inagenet database, 18 layers deep and can classify images into 1000 object categories
# here we use this to classify ants and bees
# transfer learning:
# modifying or adding a new layer
# rapid generation of new models due to the heavy time consuming models to produce newly

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),

}

#import data
data_dir = './hymenoptera_data'
sets = ['train','val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                         data_transforms[x])
                    for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=true, num_workers=4)
                    for x in ['train', 'val']}
                    