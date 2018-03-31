"""Collection of functions to write to and read from binary files."""

import tensorflow as tf
import os
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def create_binary_files(data_dir, validation_dataset_size=5000):
    """Download MNIST images, partition into training, validation and test datasets,
    and save as .tfrecords files.
    Args:
        data_dir (string): Path on the file system where binary files are created.
        validation_dataset_size (int): Number of examples to be split out from the
                                      training set to create validation set.
    """

    if not tf.gfile.Exists(data_dir):
        tf.gfile.MkDir(data_dir)

    datasets = input_data.read_data_sets(data_dir,
                                    dtype=tf.uint8,
                                    validation_size=validation_dataset_size)

    create_binary_file(datasets.train, 'train', data_dir)
    create_binary_file(datasets.validation, 'validation', data_dir)
    create_binary_file(datasets.test, 'test', data_dir)


def create_binary_file(dataset, file_name, data_dir):
    """Creates a .tfrecords file in the specified path for the given dataset.
    Args:
        dataset (object): One of training, validation or test datasets.
        file_name (string): Name of the .tfrecords file.
        data_dir (string): Path where the file must be created.
    """

    images = dataset.images
    labels = dataset.labels

    print("Saving {} examples for dataset '{}' ..".format(dataset.num_examples, file_name))

    output_filename = os.path.join(data_dir, file_name + ".tfrecords")

    with tf.python_io.TFRecordWriter(output_filename) as writer:
        for i in range(dataset.num_examples):
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'label': _int64_feature(int(labels[i])),
                        'image_raw': _bytes_feature(images[i].tobytes())
                    }
                )
            )

            writer.write(example.SerializeToString())


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def load_dataset(dataset_name, data_dir):
    """Load dataset into memory from binary file.
    Args:
        dataset_name (string): Name of the dataset.
        data_dir (string): Path of the .tfrecords file.
    """

    dataset_filename = os.path.join(data_dir, dataset_name+".tfrecords")

    images = []
    labels = []

    for serialized_example in tf.python_io.tf_record_iterator(dataset_filename):

        example = tf.train.Example()
        example.ParseFromString(serialized_example)

        label = example.features.feature['label'].int64_list.value[0]
        image_raw = example.features.feature['image_raw'].bytes_list.value[0]

        image = np.frombuffer(image_raw, dtype=np.uint8).reshape(-1)
        image = np.true_divide(image, 255.0, dtype=np.float32)

        images.append(image)
        labels.append(label)

    print("Loaded {} images.".format(len(images)))

    return images, labels


if __name__ == '__main__':
    pass
