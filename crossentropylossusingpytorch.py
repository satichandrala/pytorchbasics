import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as fnn

# def cross_entropy(actual, predicted):
#     loss = -np.sum(actual - np.log(predicted))
#     return loss

# # y must be one hot encoded in CE loss
# # imagine 3 classes 0, 1, 2
# # for correct label we put 1 and for others we give 0
# # so if class 0 is correct, y = [1, 0, 0]
# # if class 1 is correct, y = [0, 1, 0]
# # if class 2 is correct, y = [0, 0, 2]
# Y = np.array([1, 0, 0])

# # y_pred has probabilities high probabilities - low ce loss and low probs means high ce loss
# Y_pred_good = np.array([0.7, 0.2, 0.1])
# Y_pred_bad  = np.array([0.1, 0.3, 0.6])
# l1  = cross_entropy(Y, Y_pred_good)
# l2  = cross_entropy(Y, Y_pred_good)

# print(f'Loss1 numpy: {l1:.4f}')
# print(f'Loss2 numpy: {l2:.4f}')
# # results:Loss1 numpy: -5.2687
# # Loss2 numpy: -5.2687

# Using Pytorch
# No Softmax in last layer because pytorch applies nn.LogSoftmax + nn.NLLLoss (negative log likelihood loss)
# we don't one hot encode just y has class labels
# Y_pred has raw scores or logits, no Softmax

loss = nn.CrossEntropyLoss()


# loss in pytorch allows for mutliple samples
# 3 samples
Y = torch.tensor([2, 0, 1])

# # samples x nclasses = 1 x 3
# Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]]) # raw values no softr,ax applied
# Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

# samples x nclasses = 3 x 3
Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]]) # raw values no softr,ax applied
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())
# actual prediction
_, predictions1 = torch.max(Y_pred_good, 1)
_, predictions2 = torch.max(Y_pred_bad, 1)
print(predictions1)
print(predictions2)
