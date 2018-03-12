# research-eeg

Analysing data from theAmerican Epilepsy Society Seizure Prediction Challenge

https://www.kaggle.com/c/seizure-prediction/data


Data description:

Intracranial EEG (iEEG) data clips are organized in folders containing training and testing data for each human or canine subject. The training data is organized into ten minute EEG clips labeled "Preictal" for pre-seizure data segments, or "Interictal" for non-seizure data segments. Training data segments are numbered sequentially, while testing data are in random order. Within folders data segments are stored in .mat files as follow:

preictal_segment_N.mat - the Nth preictal training data segment
interictal_segment_N.mat - the Nth non-seizure training data segment
test_segment_N.mat - the Nth testing data segment
Each .mat file contains a data structure with fields as follow:

data: a matrix of EEG sample values arranged row x column as electrode x time.
data_length_sec: the time duration of each data row
sampling_frequency: the number of data samples representing 1 second of EEG data.
channels: a list of electrode names corresponding to the rows in the data field
sequence: the index of the data segment within the one hour series of clips. For example, preictal_segment_6.mat has a sequence number of 6, and represents the iEEG data from 50 to 60 minutes into the preictal data.
Preictal training and testing data segments are provided covering one hour prior to seizure with a five minute seizure horizon. (i.e. from 1:05 to 0:05 before seizure onset.) This pre-seizure horizon ensures that 1) seizures could be predicted with enough warning to allow administration of fast-acting medications, and 2) any seizure activity before the annotated onset that may have been missed by the epileptologist will not affect the outcome of the competition.
