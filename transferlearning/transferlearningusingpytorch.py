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
dataset_sizes = {x: len(image_datasets[x] for x in ['train', 'val'])}
class_names = image_datasets['train'].classes
print(class_names)
# training of the model and evaluation
def train_model(model, criterion, optimizer, scheduler, num_epochs = 25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-'*10)
# here we are doing training and validation in each epoch

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        # iterate over the data via loop, for the inputs and labels
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward loop , track history only if in training phase
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

                print('f{phase} Loss: {:.4f} Acc: {.4f}' .format(phase, epoch_loss, epoch_acc))

                # deep copy the model

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
    
            print()
        
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        print(f'Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model

model = models.resnet18(pretrained=True) # optimized and pretrained imagenet net daTA 
num_ftrs = models.fc.in_features # get input features from the last layer
# create a new layer and assign it to last layer
model.fc = nn.Linear(num_ftrs, 2) # outputs 2,we have two classes now the ants and bees classes
model.to(device) # we set the device at the top

# define loss and optimizer for the new model
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# scheduler - will update the learning rate

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size =7, gamma=0.1) # every 7 epochs will be multiplied by gamma
# loop ove the epochs
for epoch in range(100):
    train() # optimizer.step
    evaluate()


