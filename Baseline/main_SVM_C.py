from configparser import ConfigParser

import numpy as np

from quickdrawdataset import QuickDrawDataset, classes, TrainingQuickDrawLoader, ValidationQuickDrawLoader
from SVMC import SVMC
from Utils import savePredictionsToFile
import torch

from transforms import create_transforms


def loadDB(model_config, transformations):

    dataset = QuickDrawDataset(model_config.get("data_path"), split='train', transform=transformations)

    indices = torch.randperm(len(dataset))
    train_indices = indices[:len(indices) - int(model_config.getfloat("validationRatio") * len(dataset))]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_indices = indices[len(indices) - int(model_config.getfloat("validationRatio") * len(dataset)):]
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indices)

    # Data loaders
    t_loader = TrainingQuickDrawLoader(dataset, batch_size=model_config.getint("batch_size"), sampler=train_sampler,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)

    v_loader = ValidationQuickDrawLoader(dataset, batch_size=model_config.getint("batch_size"), sampler=valid_sampler,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)

    return t_loader, v_loader, dataset

## Load the config
config = ConfigParser()
config.read('models.config')
model_config = config["DEFAULT"]

transformations = create_transforms(model_config)

t_loader, v_loader, dataset = loadDB(model_config, transformations)

print("Denoising the training data and loading it in memory. This will take a while. Please be patient.")
trainX = 0
trainY = 0
for it, (x, y) in enumerate(t_loader):
    if it == 0:
        trainX = x.numpy().reshape(x.shape[0], -1)
        trainY = y.numpy()
    else:
        trainX = np.append(trainX, x.numpy().reshape(x.shape[0], -1), axis=0)
        trainY = np.append(trainY, y.numpy(), axis=0)

print("Denoising the validation data and loading it in memory. This will take a while. Please be patient.")
valX = 0
valY = 0
for it, (x, y) in enumerate(v_loader):
    if it == 0:
        valX = x.numpy().reshape(x.shape[0], -1)
        valY = y.numpy()

trainX = trainX / 255
valX = valX / 255

################## TRAIN ######################

#SVC

print("Training C (C=100) supported SVM No kernel (linear). ")
svc = SVMC(100, 'linear')
svc.train(trainX, trainY)

print("Validation score", svc.score(valX, valY))


################## TEST ######################
test_dataset = QuickDrawDataset(model_config.get("data_path"), split='test')
test_indices = torch.randperm(len(dataset))
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
t_loader = torch.utils.data.DataLoader(dataset, batch_size=model_config.getint("batch_size"), sampler=test_sampler)

print("Denoising the test data and loading it in memory. This will take a while. Please be patient.")
testX = 0
for it, (x, y) in enumerate(t_loader):
    if it == 0:
        testX = x.numpy().reshape(x.shape[0], -1)

testX = testX / 255

print("Predicting on the test set...")
testY = svc.predict(testX)

savePredictionsToFile("SVM_C_set_prediction.csv", testY, classes)

print("Predictions saved into: SVM_C_set_prediction.csv")