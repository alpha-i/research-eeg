import scipy.io as spio
import numpy as np

from providers import TrainDataProvider

DATA_PATH = '/mnt/pika/Kaggle/Data/EEG/Dog_1/'
MATLAB_EXTENSION = '.mat'
N_TRAIN_SEGMENTS = 100
N_SENSORS = 16


def make_eeg_data_provider(feature_length=100, batch_size=200):
    """ Creates trainng data of dmension [samples, series, feature_length]

    :param feature_length:
    :param batch_size:
    :return:
    """
    all_segments = load_training_segments()

    full_data = np.stack(all_segments)

    # Normalise data
    full_data = full_data - np.mean(full_data.flatten())
    full_data = full_data / np.std(full_data.flatten())

    train_x = np.reshape(full_data, (-1, feature_length))
    train_y = None

    return TrainDataProvider(train_x, train_y, batch_size)


def load_training_segments():
    """ Returns a list of segments, each a 2D nparray of dimensions [sensors=16, timesteps=239766]. """

    sensory_list = []
    for i in range(N_TRAIN_SEGMENTS):
        segment_data = load_segment(i+1)
        sensory_list.append(segment_data)

    return sensory_list


def load_segment(segment_number):
    """

    :param int segment_number: Which sample to load
    :return: 2D nparray of dimensions [sensors=16, timesteps=239766]
    """

    datafile = load_filename(segment_number)
    try:
        eeg_mat = spio.loadmat(datafile, squeeze_me=True)
        key = "interictal_segment_" + str(segment_number)  # interictal or preictal
        eeg_segment = eeg_mat[key]
        sensory_data = eeg_segment.ravel()[0][0]  # data consists of 16x239766 entries
        print("Loaded segment", segment_number, " Filename:", datafile)
    except:
        sensory_data = []
        print("Failed to load segment", segment_number, " Filename:", datafile)

    return sensory_data


def load_filename(segment_number=1):
    """

    :param segment_number:
    :return:
    """

    segment_string = str(segment_number).zfill(4)
    filename = DATA_PATH + 'Dog_1_interictal_segment_' + segment_string + MATLAB_EXTENSION
    return filename
