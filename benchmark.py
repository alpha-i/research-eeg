#  Tests performance of various anomaly detection methods
import numpy as np
from svm import ScikitSvm
from sklearn.metrics import roc_auc_score

import matlab_data as data

DEFAULT_METHOD = 'SVM'  # options: SVM, gan, ; to be implemented:
DEFAULT_FEATURE_LENGTH = 128
N_TEST_SEGMENTS = 20
DO_FFT = False

# Provisional Results
## No FFT 200 training segments
#  With 20 tests -> ROC of 0.68
## With FFT
# 100 training segments
# With 10 tests -> ROC of 0.64; With 20 tests -> ROC of 0.682

def run_eeg_performance_benchmark(method=DEFAULT_METHOD, feature_length=DEFAULT_FEATURE_LENGTH):
    """ High level function which runs both train and predict, then checks performance

    :param method: Which algorithm to test
    :return:
    """

    detector = load_detector(method, feature_length)

    print('Loading training data')
    brainwave_segments = data.load_training_segments(feature_length)

    print('Processing training data')
    training_data = data.process_segment_list(brainwave_segments, feature_length, DO_FFT)

    print('Initiating training')
    detector.train(training_data)

    sum_prediction_normal = np.zeros(N_TEST_SEGMENTS)
    sum_prediction_abnormal = np.zeros(N_TEST_SEGMENTS)
    mean_prediction_normal = np.zeros(N_TEST_SEGMENTS)
    mean_prediction_abnormal = np.zeros(N_TEST_SEGMENTS)

    for i in range(N_TEST_SEGMENTS):
        print('Loading test data', i+1, 'of', N_TEST_SEGMENTS)
        normal_test_batch = data.load_normal_test_segment(DO_FFT, feature_length=feature_length, segment_number=i)
        abnormal_test_batch = data.load_abnormal_test_segment(DO_FFT, feature_length=feature_length, segment_number=i)

        print('Assessing test data')
        prediction_normal = detector.predict(normal_test_batch)
        prediction_abnormal = detector.predict(abnormal_test_batch)

        sum_prediction_normal[i] = np.sum(prediction_normal > -1)
        sum_prediction_abnormal[i] = np.sum(prediction_abnormal > -1)
        mean_prediction_normal[i] = np.mean(prediction_normal)
        mean_prediction_abnormal[i] = np.mean(prediction_abnormal)

    make_segment_performance_summary(sum_prediction_normal, sum_prediction_abnormal)
    make_segment_performance_summary(mean_prediction_normal, mean_prediction_abnormal)


def load_detector(method=DEFAULT_METHOD, feature_length=DEFAULT_FEATURE_LENGTH):

    if method == 'SVM':
        detector = ScikitSvm()
    else:
        raise NotImplementedError('Unsupported method', method)

    return detector

def make_segment_performance_summary(predictions_normal, predictions_abnormal):
    """ Assess performance of the prediction method"""

    n_normal = len(predictions_normal)

    print("Normal segments:", predictions_normal)
    print("Abnormal segments:", predictions_abnormal)

    combined_scores = np.stack((predictions_normal, predictions_abnormal)).flatten()
    ranks = np.argsort(combined_scores)
    print("Rankings:", ranks)

    truth = np.zeros(len(combined_scores))
    truth[0:n_normal] = 1
    roc_score = roc_auc_score(truth, combined_scores)
    print("ROC Score:", roc_score)  # 0.64 as of 13/04/2018


run_eeg_performance_benchmark()
