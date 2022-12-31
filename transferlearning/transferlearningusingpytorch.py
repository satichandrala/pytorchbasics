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
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
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
# Epoch 0/19
# ----------
# val Loss: 0.0191 Acc: 0.0131

# val Loss: 0.0372 Acc: 0.0261

# val Loss: 0.0548 Acc: 0.0392

# val Loss: 0.0801 Acc: 0.0458

# val Loss: 0.1022 Acc: 0.0588

# val Loss: 0.1168 Acc: 0.0784

# val Loss: 0.1386 Acc: 0.0915

# val Loss: 0.1604 Acc: 0.1046

# val Loss: 0.1794 Acc: 0.1111

# val Loss: 0.1880 Acc: 0.1373

# val Loss: 0.2026 Acc: 0.1569

# val Loss: 0.2089 Acc: 0.1830

# val Loss: 0.2364 Acc: 0.1830

# val Loss: 0.2517 Acc: 0.1961

# val Loss: 0.2650 Acc: 0.2222

# val Loss: 0.2815 Acc: 0.2353

# val Loss: 0.2979 Acc: 0.2484

# val Loss: 0.3104 Acc: 0.2745

# val Loss: 0.3232 Acc: 0.2941

# val Loss: 0.3372 Acc: 0.3137

# val Loss: 0.3513 Acc: 0.3333

# val Loss: 0.3675 Acc: 0.3529

# val Loss: 0.3802 Acc: 0.3725

# val Loss: 0.4026 Acc: 0.3791

# val Loss: 0.4162 Acc: 0.4052

# val Loss: 0.4292 Acc: 0.4248

# val Loss: 0.4436 Acc: 0.4444

# val Loss: 0.4583 Acc: 0.4641

# val Loss: 0.4751 Acc: 0.4771

# val Loss: 0.4916 Acc: 0.4902

# val Loss: 0.5029 Acc: 0.5098

# val Loss: 0.5167 Acc: 0.5294

# val Loss: 0.5352 Acc: 0.5425

# val Loss: 0.5515 Acc: 0.5556

# val Loss: 0.5723 Acc: 0.5621

# val Loss: 0.5843 Acc: 0.5882

# val Loss: 0.5952 Acc: 0.6144

# val Loss: 0.6086 Acc: 0.6405

# val Loss: 0.6145 Acc: 0.6405

# Training complete in 0m 14s
# Best val Acc 0.640523
# Epoch 0/24
# ----------
# val Loss: 0.0164 Acc: 0.0196

# val Loss: 0.0389 Acc: 0.0196

# val Loss: 0.0626 Acc: 0.0327

# val Loss: 0.0840 Acc: 0.0458

# val Loss: 0.1070 Acc: 0.0523

# val Loss: 0.1367 Acc: 0.0588

# val Loss: 0.1589 Acc: 0.0719

# val Loss: 0.1840 Acc: 0.0784

# val Loss: 0.2150 Acc: 0.0784

# val Loss: 0.2432 Acc: 0.0915

# val Loss: 0.2566 Acc: 0.1111

# val Loss: 0.2732 Acc: 0.1307

# val Loss: 0.2844 Acc: 0.1503

# val Loss: 0.2994 Acc: 0.1699

# val Loss: 0.3208 Acc: 0.1830

# val Loss: 0.3427 Acc: 0.2026

# val Loss: 0.3628 Acc: 0.2092

# val Loss: 0.3898 Acc: 0.2222

# val Loss: 0.4161 Acc: 0.2353

# val Loss: 0.4352 Acc: 0.2484

# val Loss: 0.4581 Acc: 0.2614

# val Loss: 0.4794 Acc: 0.2745

# val Loss: 0.5023 Acc: 0.2810

# val Loss: 0.5200 Acc: 0.2941

# val Loss: 0.5375 Acc: 0.3072

# val Loss: 0.5548 Acc: 0.3268

# val Loss: 0.5754 Acc: 0.3399

# val Loss: 0.5948 Acc: 0.3464

# val Loss: 0.6066 Acc: 0.3725

# val Loss: 0.6222 Acc: 0.3922

# val Loss: 0.6478 Acc: 0.4052

# val Loss: 0.6702 Acc: 0.4183

# val Loss: 0.6886 Acc: 0.4314

# val Loss: 0.7080 Acc: 0.4379

# val Loss: 0.7282 Acc: 0.4510

# val Loss: 0.7450 Acc: 0.4706

# val Loss: 0.7648 Acc: 0.4902

# val Loss: 0.7876 Acc: 0.4967

# val Loss: 0.7920 Acc: 0.5033

# Training complete in 0m 15s
# Best val Acc 0.503268