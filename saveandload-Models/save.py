import torch
import torch.nn as nn

## COMPLETE Model ###

torch.save(model, PATH) 

#model must be defined somewhere 
model = torch.load(PATH) # setting up model with the file name
model.eval() # we set up the model to evaluation mode

# simple strategy and cons are serialized data are bound to specific classes


## recommended
## STATE DICT
torch.save(model.state_dict(), PATH)

## model must be created again with parameters

model = Model(*args, **kwargs) # create a model object
model.load_state_dict(torch.load(PATH)) # loaded_state_dict takes the loaded dictionary
model.eval()