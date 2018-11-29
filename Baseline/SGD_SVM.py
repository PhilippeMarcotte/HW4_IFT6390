import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import RBFSampler


class SGD_SVM:

    def __init__(self, useRBFKernel):
        self.rbf_feature = RBFSampler(gamma=1, random_state=1)
        self.classifier = SGDClassifier()
        self.useRBFKernel = useRBFKernel

    def batch_train(self, training_data_loader, epochs):

        classes = self.get_all_classes(training_data_loader)

        for j in range(epochs):
            #FIT
            for i, (x, y) in enumerate(training_data_loader):
                x = x.numpy()
                y = y.numpy()

                x = x.reshape(x.shape[0], -1) #flatten the data points

                x = x / 255

                if self.useRBFKernel:
                    x = self.rbf_feature.fit_transform(x)

                self.classifier.partial_fit(x, y, classes)

    def train(self, x, y, epochs):
        #param_grid = {'max_iter': [10, 100, 1000], 'alpha': [0.0001, 0.001, 0.01, 0.1]}

        #self.classifier = GridSearchCV(SGDClassifier(n_jobs = -1, early_stopping = True), param_grid)

        self.classifier = SGDClassifier(max_iter = epochs, n_jobs = -1, early_stopping = True)

        if self.useRBFKernel:
            x = self.rbf_feature.fit_transform(x)

        self.classifier.fit(x, y)

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
            x = x.numpy()
            y = y.numpy()

            x = x.reshape(x.shape[0], -1) #flatten the data points

            x = x / 255

            if self.useRBFKernel:
                x = self.rbf_feature.fit_transform(x)

            score = self.score(x, y)
            scores_sum += score
            nbr_batches = i + 1

        return scores_sum / nbr_batches
