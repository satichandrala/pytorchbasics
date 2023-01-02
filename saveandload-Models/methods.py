import torch
import torch.nn as nn

torch.save(arg, PATH) # torch.save uses pickle module to serialize the objects and save them and the result is serialized objects which are not human readable.

torch.load(PATH)

model.load_state_dict(arg)