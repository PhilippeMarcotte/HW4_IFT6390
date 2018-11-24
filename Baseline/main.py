from quickdrawdataset import QuickDrawDataset
from Baseline.SVM import SVM
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

svm = SVM()

svm.train(t_loader, 10)

print(svm.batch_score(v_loader))