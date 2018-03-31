"""Collection of functions to do processing on input data."""

import os
import numpy as np
import tensorflow as tf

from . import io


def get_training_inputs(batch_size, data_dir):
    """Returns the input function for reading training data.
    Args:
        batch_size (int): The batch size
        data_dir (string): The path to the .tfrecords file that contains training examples
    """

    dataset_iterator_initializer_hook = DatasetIteratorInitializerRunHook()

    training_input_filename = os.path.join(data_dir, "train.tfrecords")

    def training_input_fn():

        file_names = tf.placeholder(tf.string, shape=[None], name="file_names")
        iterator = get_batch_iterator(file_names, batch_size)

        dataset_iterator_initializer_hook.set_iterator_initializer_func(

            lambda sess: sess.run(iterator.initializer, feed_dict={
                file_names: [training_input_filename]
            })
        )

        X, y = iterator.get_next()

        return {'X': X}, y

    return training_input_fn, dataset_iterator_initializer_hook


def get_test_inputs(batch_size, data_dir):
    """Returns an input function to load the test dataset as numpy arrays of images and labels.
    Args:
        batch_size (int): Size of the examples batch
        data_dir (string): The path to the .tfrecords file that contains training examples
    """

    test_images, test_labels = io.load_dataset("test", data_dir)

    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return tf.estimator.inputs.numpy_input_fn(
        x={'X': test_images},
        y=test_labels,
        num_epochs=1,
        batch_size=batch_size,
        shuffle=False
    )


def get_batch_iterator(file_names, batch_size):
    """Creates a dataset iterator over a set of TFRecords files.
    Args:
        file_names (object): A placeholder for filenames tensor
        batch_size (int): Size of the examples batch
    """

    dataset = tf.data.TFRecordDataset(file_names) \
        .map(_example_proto_to_features_fn, num_parallel_calls=10) \
        .map(_scale_image_fn, num_parallel_calls=20) \
        .shuffle(buffer_size=2000) \
        .repeat() \
        .batch(batch_size) \
        .prefetch(1)    # make sure there is always 1 batch ready to be served

    return dataset.make_initializable_iterator()


def _example_proto_to_features_fn(example_proto):

    features = tf.parse_single_example(example_proto, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)

    image = tf.reshape(image, [28 * 28])

    return image, label


def _scale_image_fn(image, label):

    scaled_image = tf.multiply(tf.cast(image, tf.float32), 1.0 / 255.0)
    return scaled_image, label


class DatasetIteratorInitializerRunHook(tf.train.SessionRunHook):

    def __init__(self):
        super(DatasetIteratorInitializerRunHook, self).__init__()
        self.iterator_initializer_func = None

    def set_iterator_initializer_func(self, func):
        self.iterator_initializer_func = func

    def after_create_session(self, session, coord):
        self.iterator_initializer_func(session)


if __name__ == '__main__':
    pass
