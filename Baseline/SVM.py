import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler

class SVM:

    def __init__(self):
        self.rbf_feature = RBFSampler(gamma=1, random_state=1)
        self.classifier = SGDClassifier()

    def train(self, training_data_loader, epochs):


        classes = self.get_all_classes(training_data_loader)

        for j in range(epochs):
            #FIT
            for i, (x, y) in enumerate(training_data_loader):
                print("training batch: ", i)
                #Apply kernel transformation
                x = x.numpy()
                y = y.numpy()

                x = x.reshape(x.shape[0], -1) #flatten the data points
                transformed_X = self.rbf_feature.fit_transform(x)

                self.classifier.partial_fit(transformed_X, y, classes)

    def get_all_classes(self, training_data_loader):
        classes = []
        for i, (x, y) in enumerate(training_data_loader):
            classes.append(y.numpy())
        classes = np.unique(classes)
        return classes

    def predict(self, x):
        return self.classifier.predict(x)

    def score(self, x, y):
        return self.classifier.score(x, y)

    def batch_score(self, validation_data_loader):

        nbr_batches = 0
        scores_sum = 0

        for i, (x, y) in enumerate(validation_data_loader):
            print("scoring batch: ", i)
            x = x.numpy()
            y = y.numpy()

            #Apply kernel transformation
            x = x.reshape(x.shape[0], -1) #flatten the data points
            transformed_X = self.rbf_feature.fit_transform(x)

            score = self.score(transformed_X, y)
            scores_sum += score
            nbr_batches = i + 1

        return scores_sum / nbr_batches
