import os
import tensorflow as tf

import conf


def get_sample_datasets():
    # Training samples path, change to your local path
    training_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                                                         os.path.join(conf.data_directory, "sampledata/trainingSamples.csv"))
    # Test samples path, change to your local path
    test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
                                                     os.path.join(conf.data_directory, "sampledata/testSamples.csv"))

    # load sample as tf dataset
    def get_dataset(file_path):
        dataset = tf.data.experimental.make_csv_dataset(
            file_path,
            batch_size=12,
            label_name='label',
            na_value="0",
            num_epochs=1,
            ignore_errors=True)
        return dataset

    # split as test dataset and training dataset
    train_dataset = get_dataset(training_samples_file_path)
    test_dataset = get_dataset(test_samples_file_path)
    return train_dataset, test_dataset
