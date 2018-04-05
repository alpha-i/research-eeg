#  Tests performance of various anomaly detection methods
import numpy as np
from svm import ScikitSvm

import matlab_data as data

DEFAULT_METHOD = 'SVM'
DEFAULT_FEATURE_LENGTH = 100


def run_eeg_performance_benchmark(method=DEFAULT_METHOD, feature_length=DEFAULT_FEATURE_LENGTH):
    """ High level function which runs both train and predict, then checks performance

    :param method: Which algorithm to test
    :return:
    """

    detector = load_detector(method, feature_length)

    brainwave_segments = data.load_training_segments(feature_length)
    training_data = data.process_segment_list(brainwave_segments, feature_length)

    normal_test_batch = data.load_normal_test_batch(feature_length=feature_length)
    abnormal_test_batch = data.load_abnormal_test_batch(feature_length=feature_length)

    print('Initiating training')
    detector.train(training_data)

    print('Assessing test data')
    prediction_normal = detector.predict(normal_test_batch)
    prediction_abnormal = detector.predict(abnormal_test_batch)

    make_performance_summary(prediction_normal, prediction_abnormal)


def load_detector(method=DEFAULT_METHOD, feature_length=DEFAULT_FEATURE_LENGTH):

    if method == 'SVM':
        detector = ScikitSvm()
    else:
        raise NotImplementedError('Unsupported method', method)

    return detector


def make_performance_summary(prediction_normal, prediction_abnormal):
    """ Assess predictive performance. """

    n_norm = len(prediction_normal)
    n_abnorm = len(prediction_abnormal)

    norm_success_rate = - np.mean(prediction_normal)
    abnormal_success_rate = np.mean(prediction_abnormal) + 1

    fraction_false_pos = prediction_normal[prediction_normal != -1].size / n_norm
    fraction_false_neg = prediction_abnormal[prediction_abnormal == -1].size / n_abnorm

    print("False positive rate:", fraction_false_pos)
    print("False negative rate", fraction_false_neg)




run_eeg_performance_benchmark()
