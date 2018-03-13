from abc import ABCMeta, abstractmethod
from collections import namedtuple
import logging

import numpy as np
from alphai_data_sources.data_sources import DataSourceGenerator
from alphai_data_sources.generator import BatchGenerator, BatchOptions

logging.getLogger(__name__).addHandler(logging.NullHandler())

TrainData = namedtuple('TrainData', 'features labels')


class AbstractTrainDataProvider(metaclass=ABCMeta):

    _batch_size = None

    @property
    @abstractmethod
    def n_train_samples(self):
        raise NotImplementedError

    @abstractmethod
    def shuffle_data(self):
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, batch_number):
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def number_of_batches(self):
        return int(np.ceil(self.n_train_samples / self._batch_size))


class TrainDataProvider(AbstractTrainDataProvider):
    def __init__(self, features, labels, batch_size):
        self._train_data = TrainData(features, labels)
        self._batch_size = batch_size
        self._noise_data = None

    @property
    def n_train_samples(self):
        return self._train_data.features.shape[0]

    def shuffle_data(self):
        """ Reorder the numpy arrays in a random but consistent manner """

        features = self._train_data.features
        labels = self._train_data.labels

        rng_state = np.random.get_state()
        np.random.shuffle(features)
        np.random.set_state(rng_state)
        if labels is not None:
            np.random.shuffle(labels)

        self._train_data = TrainData(features, labels)

    def get_batch(self, batch_number):
        """ Returns batch of features and labels from the full data set x and y

        :param nparray x: Full set of training features
        :param nparray y: Full set of training labels
        :param int batch_number: Which batch
        :param batch_size:
        :return:
        """
        features = self._train_data.features
        labels = self._train_data.labels

        lo_index = batch_number * self.batch_size
        hi_index = lo_index + self.batch_size
        batch_features = features[lo_index:hi_index, :]

        if labels is None:
            batch_labels = None
        else:
            batch_labels = labels[lo_index:hi_index, :]

        return TrainData(batch_features, batch_labels)

    def get_noisy_batch(self, batch_number, noise_amplitude=0):
        """ Returns noisy batches and original labels from the full data set x and y

        :param batch_number:
        :param noise_amplitude:
        :return:
        """
        batch = self.get_batch(batch_number)

        if noise_amplitude > 0:
            noise_batch = self._get_noise(batch_number)
            noisy_batch_features = batch.features + noise_batch * noise_amplitude
            batch = TrainData(noisy_batch_features, batch.labels)

        return batch

    def _get_noise(self, batch_number):
        """ Returns fixed noise associated with batch number. Could also use rand seed but this saves recomputing.

        :param batch_number:
        :return:
        """
        if self._noise_data is None:
            self.initialise_noise()
        noise = self._noise_data
        lo_index = batch_number * self.batch_size
        hi_index = lo_index + self.batch_size

        return noise[lo_index:hi_index, :]

    def initialise_noise(self):
        """ Create noise to match train data

        :return:
        """

        noise_shape = self._train_data.features.shape
        self._noise_data = np.random.normal(loc=0.0, scale=1.0, size=noise_shape)


class TrainDataProviderForDataSource(AbstractTrainDataProvider):

    def __init__(self, series_name, dtype, n_train_samples, batch_size, for_training,  bin_edges=None):
        self._batch_size = batch_size
        self._batch_generator = BatchGenerator()
        self._n_train_samples = n_train_samples
        self._bin_edges = bin_edges
        self._dtype = dtype
        self._for_training = for_training
        self._noise_data = None

        data_source_generator = DataSourceGenerator()
        self._data_source = data_source_generator.make_data_source(series_name)

    def get_batch(self, batch_number):
        batch_options = BatchOptions(self._batch_size, batch_number, self._for_training, self._dtype)

        features, labels = self._batch_generator.get_batch(batch_options, self._data_source)

        if self._bin_edges is not None:
            labels = classify_labels(self._bin_edges, labels)

        features = np.swapaxes(features, axis1=1, axis2=2)
        labels = np.swapaxes(labels, axis1=1, axis2=2)

        # Single channel
        features = np.expand_dims(features, axis=1)

        if self._bin_edges is None:
            labels = np.expand_dims(labels, axis=-1)
            labels = np.swapaxes(labels, axis1=1, axis2=-1)

        return TrainData(features, labels)

    def get_noisy_batch(self, batch_number, noise_amplitude=0):
        """ Returns noisy batches and original labels from the full data set x and y

        :param batch_number:
        :param noise_amplitude:
        :return:
        """
        batch = self.get_batch(batch_number)

        if noise_amplitude > 0:
            feature_factor = 1 / (1 + noise_amplitude)
            noise_factor = noise_amplitude / (1 + noise_amplitude)

            noise_batch = self._get_noise(batch_number)
            noisy_batch_features = batch.features * feature_factor + noise_batch * noise_factor
            batch = TrainData(noisy_batch_features, batch.labels)

        return batch

    def _get_noise(self, batch_number):
        """ Returns fixed noise associated with batch number. Could also use rand seed but this saves recomputing.

        :param batch_number:
        :return:
        """
        if self._noise_data is None:
            self.initialise_noise()

        lo_index = batch_number * self.batch_size
        hi_index = lo_index + self.batch_size

        return self._noise_data[lo_index:hi_index, :]

    def initialise_noise(self):
        """ Create noise to match train data. Currently only mnist is supported

        :return:
        """

        if self._for_training:
            noise_shape = (60000, 1, 28, 28)
        else:
            noise_shape = (10000, 1, 28, 28)

        mnist_sigma = 0.31
        self._noise_data = np.random.normal(loc=0.0, scale=mnist_sigma, size=noise_shape)

    @property
    def n_train_samples(self):
        return self._n_train_samples

    def shuffle_data(self):
        pass

    @property
    def data_source(self):
        return self._data_source
