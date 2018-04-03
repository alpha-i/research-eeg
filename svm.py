# Standard procedure for anomaly detection

from sklearn import svm


class ScikitSvm:

    def __init__(self):
        self.classifier = svm.SVC()

    def train(self, train_data):

        self.classifier.fit(train_data)

    def predict(self, predict_data):

        return self.classifier.predict(predict_data)
