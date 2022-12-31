# pretrained ResNet18 CNN, trained on more than a million images from inagenet database, 18 layers deep and can classify images into 1000 object categories
# here we use this to classify ants and bees
# transfer learning:
# modifying or adding a new layer
# rapid generation of new models due to the heavy time consuming models to produce newly

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
}

#import data
data_dir = "C:/Users/satic/Desktop/Uni_Bonn/Master_Thesis-UniBonn/code/pytorchbasics/transferlearning/hymenoptera_data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), 
                                            data_transforms[x]) 
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                            shuffle=True, num_workers=0) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(class_names)

def imshow(inp, title):
    ''' ImShow for Tensor.'''
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()

# get a bacth of training data

inputs, classes = next(iter(dataloaders['train']))

# make a grid from the batch

out = torchvision.utils.make_grid(inputs)

imshow(out, title=[class_names[x] for x in classes])

# training of the model and evaluation
def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-'*10)
# here we are doing training and validation in each epoch

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # set model to training mode
            else:
                model.eval() # set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over the data via loop, for the inputs and labels
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

            # forward loop , 
            # track history only if in training phase

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                # backward and optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}' .format(phase, epoch_loss, epoch_acc))

                # deep copy the model

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best val Acc {best_acc:4f}')

        # load best model weights
    model.load_state_dict(best_model_wts)
    return model
####
# we use the technique called finetuning. by finetuning all the weights based on the data
# We start transfer learning 
# #### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.

model = models.resnet18(pretrained=True) # optimized and pretrained imagenet net daTA 
# for param in model.parameters():
#     params.requires_grad = False

num_ftrs = model.fc.in_features
 # get input features from the last layer
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
# create a new layer and assign it to last layer
model.fc = nn.Linear(num_ftrs, 2) # outputs 2,we have two classes now the ants and bees classes
model.to(device) # we set the device at the top

# define loss and optimizer for the new model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001) # optimization module, SGD to optimize the model

# scheduler - will update the learning rate

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1) # every 7 epochs will be multiplied by gamma to update the learning rate i.e., 10%

# loop over the epochs - generally

# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
#### ConvNet as fixed feature extractor ####
# Here, we need to freeze all the network except the final layer.
# We need to set requires_grad == False to freeze the parameters so that the gradients are not computed in backward()

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=20)

# we use the technique called finetuning. by finetuning all the weights based on the data
# We start transfer learning

model_conv = torchvision.models.resnet18(pretrained=True) # optimized and pretrained imagenet net daTA 

for param in model_conv.parameters():
    param.requires_grad = False

num_ftrs = model_conv.fc.in_features # get input features from the last layer
# create a new layer and assign it to last layer
model_conv.fc = nn.Linear(num_ftrs, 2) # outputs 2,we have two classes now the ants and bees classes
model_conv.to(device) # we set the device at the top

# define loss and optimizer for the new model
criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr = 0.001, momentum=0.9) # optimization module, SGD to optimize the model

# scheduler - will update the learning rate

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size =7, gamma=0.1) # every 7 epochs will be multiplied by gamma to update the learning rate i.e., 10%

# loop over the epochs - generally

# for epoch in range(100):
#     train() # optimizer.step
#     evaluate()
#     scheduler.step()


model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)


# Output

# Epoch 1/19
# ----------
# train Loss: 0.5337 Acc: 0.7582
# val Loss: 0.3582 Acc: 0.9020

# Epoch 2/19
# ----------
# train Loss: 0.4683 Acc: 0.7828
# val Loss: 0.2885 Acc: 0.9150

# Epoch 3/19
# ----------
# train Loss: 0.3921 Acc: 0.8238
# val Loss: 0.2565 Acc: 0.9281

# Epoch 4/19
# ----------
# train Loss: 0.4115 Acc: 0.8074
# val Loss: 0.2350 Acc: 0.9281

# Epoch 5/19
# ----------
# train Loss: 0.3664 Acc: 0.8566
# val Loss: 0.2472 Acc: 0.9346

# Epoch 6/19
# ----------
# train Loss: 0.4547 Acc: 0.8238
# val Loss: 0.2263 Acc: 0.9346

# Epoch 7/19
# ----------
# train Loss: 0.3744 Acc: 0.8320
# val Loss: 0.2330 Acc: 0.9281

# Epoch 8/19
# ----------
# train Loss: 0.3586 Acc: 0.8402
# val Loss: 0.2316 Acc: 0.9346

# Epoch 9/19
# ----------
# train Loss: 0.3589 Acc: 0.8320
# val Loss: 0.2200 Acc: 0.9412

# Epoch 10/19
# ----------
# train Loss: 0.3775 Acc: 0.8443
# val Loss: 0.2277 Acc: 0.9346

# Epoch 11/19
# ----------
# train Loss: 0.3013 Acc: 0.8975
# val Loss: 0.2201 Acc: 0.9346

# Epoch 12/19
# ----------
# train Loss: 0.3018 Acc: 0.8730
# val Loss: 0.2230 Acc: 0.9346

# Epoch 13/19
# ----------
# train Loss: 0.4123 Acc: 0.8238
# val Loss: 0.2129 Acc: 0.9346

# Epoch 14/19
# ----------
# train Loss: 0.3616 Acc: 0.8484
# val Loss: 0.2391 Acc: 0.9281

# Epoch 15/19
# ----------
# train Loss: 0.3708 Acc: 0.8402
# val Loss: 0.2259 Acc: 0.9281

# Epoch 16/19
# ----------
# train Loss: 0.3505 Acc: 0.8566
# val Loss: 0.2183 Acc: 0.9281

# Epoch 17/19
# ----------
# train Loss: 0.3802 Acc: 0.8320
# val Loss: 0.2362 Acc: 0.9346

# Epoch 18/19
# ----------
# train Loss: 0.2907 Acc: 0.8893
# val Loss: 0.2303 Acc: 0.9281

# Epoch 19/19
# ----------
# train Loss: 0.3216 Acc: 0.8811
# val Loss: 0.2183 Acc: 0.9346

# Training complete in 23m 59s
# Best val Acc 0.941176
# Epoch 0/24
# ----------
# train Loss: 0.6005 Acc: 0.7131
# val Loss: 0.2137 Acc: 0.9281

# Epoch 1/24
# ----------
# train Loss: 0.4646 Acc: 0.7787
# val Loss: 0.1940 Acc: 0.9412

# Epoch 2/24
# ----------
# train Loss: 0.4322 Acc: 0.8197
# val Loss: 0.2583 Acc: 0.9020

# Epoch 3/24
# ----------
# train Loss: 0.5690 Acc: 0.7582
# val Loss: 0.1749 Acc: 0.9412

# Epoch 4/24
# ----------
# train Loss: 0.7413 Acc: 0.7377
# val Loss: 1.2826 Acc: 0.6013

# Epoch 5/24
# ----------
# train Loss: 0.6218 Acc: 0.7746
# val Loss: 0.2379 Acc: 0.9216

# Epoch 6/24
# ----------
# train Loss: 0.3084 Acc: 0.8566
# val Loss: 0.2141 Acc: 0.9216

# Epoch 7/24
# ----------
# train Loss: 0.3777 Acc: 0.8156
# val Loss: 0.1893 Acc: 0.9412

# Epoch 8/24
# ----------
# train Loss: 0.3266 Acc: 0.8484
# val Loss: 0.1746 Acc: 0.9477

# Epoch 9/24
# ----------
# train Loss: 0.4545 Acc: 0.8197
# val Loss: 0.1581 Acc: 0.9412

# Epoch 10/24
# ----------
# train Loss: 0.3241 Acc: 0.8648
# val Loss: 0.1909 Acc: 0.9346

# Epoch 11/24
# ----------
# train Loss: 0.3793 Acc: 0.8197
# val Loss: 0.1782 Acc: 0.9346

# Epoch 12/24
# ----------
# train Loss: 0.2313 Acc: 0.9221
# val Loss: 0.1757 Acc: 0.9412

# Epoch 13/24
# ----------
# train Loss: 0.3656 Acc: 0.8484
# val Loss: 0.2034 Acc: 0.9346

# Epoch 14/24
# ----------
# train Loss: 0.3348 Acc: 0.8525
# val Loss: 0.1630 Acc: 0.9477

# Epoch 15/24
# ----------
# train Loss: 0.4392 Acc: 0.8156
# val Loss: 0.1615 Acc: 0.9542

# Epoch 16/24
# ----------
# train Loss: 0.3480 Acc: 0.8566
# val Loss: 0.1654 Acc: 0.9477

# Epoch 17/24
# ----------
# train Loss: 0.3686 Acc: 0.8361
# val Loss: 0.1900 Acc: 0.9281

# Epoch 18/24
# ----------
# train Loss: 0.3655 Acc: 0.8197
# val Loss: 0.1722 Acc: 0.9542

# Epoch 19/24
# ----------
# train Loss: 0.3316 Acc: 0.8648
# val Loss: 0.1661 Acc: 0.9477

# Epoch 20/24
# ----------
# train Loss: 0.3319 Acc: 0.8607
# val Loss: 0.1744 Acc: 0.9346

# Epoch 21/24
# ----------
# train Loss: 0.3007 Acc: 0.8730
# val Loss: 0.1680 Acc: 0.9542

# Epoch 22/24
# ----------
# train Loss: 0.2947 Acc: 0.8811
# val Loss: 0.1719 Acc: 0.9542

# Epoch 23/24
# ----------
# train Loss: 0.3496 Acc: 0.8525
# val Loss: 0.1920 Acc: 0.9346

# Epoch 24/24
# ----------
# train Loss: 0.2980 Acc: 0.8730
# val Loss: 0.1696 Acc: 0.9477

# Training complete in 15m 18s
# Best val Acc 0.954248