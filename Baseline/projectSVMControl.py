import numpy as np

from quickdrawdataset import QuickDrawDataset
from Baseline.svm_project import SVMC
import torch

data_path = "../data"
validationRatio = 0.1
batch_size = 100

dataset = QuickDrawDataset(data_path, split='train')

indices = torch.randperm(len(dataset))
train_indices = indices[:len(indices) - int(validationRatio * len(dataset))]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
valid_indices = indices[len(indices) - int(validationRatio * len(dataset)):]
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

# Data loaders
t_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,
                                           pin_memory=(torch.cuda.is_available()), num_workers=0)

v_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,)

trainX = 0
trainY = 0
for it, (x, y) in enumerate(t_loader):
    if it == 0:
        trainX = x.numpy().reshape(x.shape[0], -1)
        trainY = y.numpy()
    else:
        trainX = np.append(trainX, x.numpy().reshape(x.shape[0], -1), axis=0)
        trainY = np.append(trainY, y.numpy(), axis=0)


valX = 0
valY = 0
for it, (x, y) in enumerate(v_loader):
    if it == 0:
        valX = x.numpy().reshape(x.shape[0], -1)
        valY = y.numpy()
    else:
        valX = np.append(valX, x.numpy().reshape(x.shape[0], -1), axis=0)
        valY = np.append(valY, y.numpy(), axis=0)


svc = SVMC(10, 'rbf')

svc.train(trainX, trainY)

print(svc.score(valX, valY))