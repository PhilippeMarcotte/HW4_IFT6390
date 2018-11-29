from sklearn.svm import SVC

class SVMC:

    def __init__(self, C_value=1, kernel_value='rbf'):
        self.C_value = C_value
        self.kernel_value = kernel_value
        self.classifier = SVC(C=self.C_value, kernel=self.kernel_value, random_state=0)

    def train(self, x, y):
        self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict(x)

    def score(self, x, y):
        return self.classifier.score(x, y)

