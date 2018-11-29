import numpy as np

from Baseline.Utils import savePredictionsToFile
from Ensemble.quickdrawdataset import QuickDrawDataset, classes
from Baseline.SGD_SVM import SGD_SVM
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


svm = SGD_SVM(False)
svm.batch_train(t_loader, 1)

print("No RBF, 10 epochs", svm.batch_score(v_loader))

##################TEST######################
test_dataset = QuickDrawDataset(data_path, split='test')
test_indices = torch.randperm(len(dataset))
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)
t_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

testX = 0
for it, (x, y) in enumerate(v_loader):
    if it == 0:
        testX = x.numpy().reshape(x.shape[0], -1)
    else:
        testX = np.append(testX, x.numpy().reshape(x.shape[0], -1), axis=0)

testX = testX / 255

testY = svm.predict(testX)

savePredictionsToFile("svm_Test.csv", testY, classes)
