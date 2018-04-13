# Standard procedure for anomaly detection

import numpy as np
from sklearn import svm

MAX_N_SAMPLES = 1000  # 20k default; 1k for speed/testing
DEFAULT_NU = 0.5       # sometimes 0.1

# nu=0.1; 4k samples; 100 timesteps:
# False positive rate: 0.04823738005840634
# False negative rate 0.9395598664997914
#
# nu = 0.5 doesnt change much
#False positive rate: 0.04776804338756779
# False negative rate 0.9405506883604505
# nu = 0.01
# False positive rate: 0.04675114726741761
# False negative rate 0.941776178556529

# Now 20k samples:
# False positive rate: 0.12257509386733417
# False negative rate: 0.8603723404255319

# Now 100k samples; 200 timesteps:
# False positive rate:
# False negative rate

class ScikitSvm:

    def __init__(self, nu=DEFAULT_NU):
        self.classifier = svm.OneClassSVM(nu=nu, kernel="rbf", gamma=0.1)

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


