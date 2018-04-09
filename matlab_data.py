import scipy.io as spio
import numpy as np
import platform

from providers import TrainDataProvider

if platform.system() == 'Darwin':
    DATA_PATH = '/Users/fergus/Kaggle/EEG/Dog_1/'
else:
    DATA_PATH = '/mnt/pika/Kaggle/Data/EEG/Dog_1/'
MATLAB_EXTENSION = '.mat'
N_TRAIN_SEGMENTS = 100
N_TEST_SEGMENTS = 24
N_SENSORS = 16


def make_eeg_data_provider(feature_length=100, batch_size=200):
    """ Creates trainng data of dimension [samples, series, feature_length]

    :param feature_length:
    :param batch_size:
    :return:
    """
    all_segments = load_training_segments(feature_length)

    train_x = process_segment_list(all_segments, feature_length)
    train_y = None

    return TrainDataProvider(train_x, train_y, batch_size)


def process_segment_list(segment_list, feature_length):
    """ Prepare list of large data samples for entry into network. """

    full_data = np.stack(segment_list)
    print('stacked')

    # Normalise data
    full_data = full_data - np.mean(full_data.flatten())
    print('meaned')
    full_data = full_data / np.std(full_data.flatten())
    print('normed')

    return np.reshape(full_data, (-1, feature_length))


def load_abnormal_test_batch(feature_length):

    sensory_list = []
    for i in range(N_TEST_SEGMENTS):
        index = i + 1
        segment_data = load_segment(index, abnormal=True)
        segment_data = _trim_segment(segment_data, feature_length)
        sensory_list.append(segment_data)

    return process_segment_list(sensory_list, feature_length)


def load_normal_test_batch(feature_length):

    sensory_list = []
    for i in range(N_TEST_SEGMENTS):
        index = i + N_TRAIN_SEGMENTS
        segment_data = load_segment(index)
        segment_data = _trim_segment(segment_data, feature_length)
        sensory_list.append(segment_data)

    return process_segment_list(sensory_list, feature_length)


def load_training_segments(feature_length):
    """ Returns a list of segments, each a 2D nparray of dimensions [sensors=16, timesteps=239766]. """

    sensory_list = []
    for i in range(N_TRAIN_SEGMENTS):
        segment_data = load_segment(i+1)
        segment_data = _trim_segment(segment_data, feature_length)
        sensory_list.append(segment_data)

    return sensory_list


def _trim_segment(segment_data, feature_length):
    """ Removes end of segment to ensure integer multiples of feature_length is available. """

    len_segment = segment_data.shape[1]
    n_features = int(len_segment / feature_length)
    max_index = n_features * feature_length

    return segment_data[:, 0:max_index]


def load_segment(segment_number, abnormal=False):
    """

    :param int segment_number: Which sample to load
    :return: 2D nparray of dimensions [sensors=16, timesteps=239766]
    """

    datafile = load_filename(segment_number, abnormal)
    regime = 'preictal' if abnormal else 'interictal'

    try:
        eeg_mat = spio.loadmat(datafile, squeeze_me=True)
        key = regime + "_segment_" + str(segment_number)  # interictal or preictal
        eeg_segment = eeg_mat[key]
        sensory_data = eeg_segment.ravel()[0][0]  # data consists of 16x239766 entries
        print("Loaded segment", segment_number, " Filename:", datafile)
    except:
        sensory_data = []
        print("Failed to load segment", segment_number, " Filename:", datafile)

    return sensory_data


def load_filename(segment_number=1, abnormal=False):
    """

    :param segment_number:
    :return:
    """

    segment_string = str(segment_number).zfill(4)
    regime = 'preictal' if abnormal else 'interictal'
    filename = DATA_PATH + 'Dog_1_' + regime + '_segment_' + segment_string + MATLAB_EXTENSION
    return filename
