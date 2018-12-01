from configparser import ConfigParser

import numpy as np

from Utils import savePredictionsToFile
from quickdrawdataset import QuickDrawDataset, classes, TrainingQuickDrawLoader, ValidationQuickDrawLoader
from SGD_SVM import SGD_SVM
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

## Load the config ##
config = ConfigParser()
config.read('models.config')
model_config = config["DEFAULT"]

transformations = create_transforms(model_config)

t_loader, v_loader, dataset = loadDB(model_config, transformations)

################## TRAIN ######################
svm = SGD_SVM(False)

print("Training SVM using SGD. No kernel (linear). ", model_config.getint("epochs"), " epochs")
print("(Denoising while training.)")
svm.batch_train(t_loader, model_config.getint("epochs"))

print("Validation score", svm.batch_score(v_loader))

################## TEST ######################
test_dataset = QuickDrawDataset(model_config.get("data_path"), split='test', transform=transformations)
test_indices = torch.randperm(len(dataset))
test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_indices)
t_loader = torch.utils.data.DataLoader(dataset, batch_size=model_config.getint("batch_size"), sampler=test_sampler)

print("Denoising the test data and loading it in memory. This will take a while. Please be patient.")
testX = 0
for it, (x, y) in enumerate(t_loader):
    if it == 0:
        testX = x.numpy().reshape(x.shape[0], -1)
    else:
        testX = np.append(testX, x.numpy().reshape(x.shape[0], -1), axis=0)

testX = testX / 255

print("Predicting on the test set...")
testY = svm.predict(testX)

savePredictionsToFile("SGD_SVM_test_set_prediction.csv", testY, classes)

print("Predictions saved into: SGD_SVM_test_set_prediction.csv")
