# Standard procedure for anomaly detection

import numpy as np
from sklearn import svm

MAX_N_SAMPLES = 4000  # 1e3 samples and 1e3 features takes approx 1323 -  to train


class ScikitSvm:

    def __init__(self):
        self.classifier = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

    def train(self, train_data):

        n_train_samples = train_data.shape[0]

        if n_train_samples > MAX_N_SAMPLES:
            train_data = self._subsample_data(train_data, MAX_N_SAMPLES)

        self.classifier.fit(train_data)

    def predict(self, predict_data):

        return self.classifier.predict(predict_data)

    @staticmethod
    def _subsample_data(train_data, MAX_N_SAMPLES):

        return train_data[np.random.choice(train_data.shape[0], MAX_N_SAMPLES, replace=False)]

