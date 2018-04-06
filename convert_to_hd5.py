import glob
import warnings
import os

import h5py
import click
import scipy.io as spio

warnings.simplefilter(action='ignore', category=FutureWarning)

ABNORMAL = 'preictal'
NORMAL = 'interictal'
TEST = 'test'

DATA_TYPE_STORE_KEY_MAPPING = {
    ABNORMAL: 'ABNORMAL',
    NORMAL: 'NORMAL',
    TEST: 'TEST'
}

FILEMASK = '{}_{}_*.mat'
KEY_TEMPLATE = '{}_segment_{}'


def _read_and_parse_files(input_directory, subject_name, type_of_samples):
    filemask = FILEMASK.format(subject_name, type_of_samples)
    samples = []
    for full_file_path in glob.glob(os.path.join(input_directory, filemask)):
        click.echo("loading {} file {}".format(type_of_samples, full_file_path))

        datafile = spio.loadmat(full_file_path, squeeze_me=True)
        sample_number = os.path.splitext(os.path.basename(full_file_path))[0].split('_')[-1]
        key = KEY_TEMPLATE.format(type_of_samples, int(sample_number))
        sample_data = {
            'data': datafile[key].ravel()[0][0],
            'sample_length_seconds': datafile[key].ravel()[0][1],
            'sample_rate': datafile[key].ravel()[0][2]
        }

        samples.insert(int(sample_number), sample_data)
    return samples


def _populate_store(group_name, normal_data, store):
    group = store.create_group(group_name)
    for i, sample in enumerate(normal_data):
        subgroup = group.create_group('SAMPLE_{}'.format(str(i).zfill(4)))
        for k, value in sample.items():
            subgroup.create_dataset(k, data=value)


@click.command()
@click.argument('input_directory', type=click.Path(exists=True))
@click.argument('subject_name', type=click.STRING)
@click.argument('destination_file', type=click.STRING)
def convert(input_directory, subject_name, destination_file):

    if os.path.isfile(destination_file):
        if not click.confirm('Destination file {} exists. Do you want to continue?'.format(destination_file)):
            exit()

    store = h5py.File(destination_file, 'w')

    for type_of_samples, group_name in DATA_TYPE_STORE_KEY_MAPPING.items():
        data = _read_and_parse_files(input_directory, subject_name, type_of_samples)
        _populate_store(group_name, data, store)

    store.close()


if __name__ == '__main__':
    convert()
