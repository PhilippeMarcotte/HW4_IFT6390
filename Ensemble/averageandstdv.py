from kdataset import QuickDrawDataset, DenoisedQuickDrawDataset
import numpy as np

train = DenoisedQuickDrawDataset("../data", split="train")

test = DenoisedQuickDrawDataset("../data", split="test")

dataset=np.concatenate((train.data, test.data), axis=0)
print(dataset.mean() / 255.0, dataset.std() / 255.0)
