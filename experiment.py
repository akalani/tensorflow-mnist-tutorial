import sys
import tensorflow as tf
from tensorflow.contrib import slim

from input import io, preprocessing

flags = tf.app.flags
flags.DEFINE_integer("batch_size", 100, "Batch size")
flags.DEFINE_integer("epochs", 1, "Number of epochs.")
flags.DEFINE_string("data_dir", "/tmp/data/mnist", "Path from where to load input data")
flags.DEFINE_string("model_dir", "/tmp/ws/mnist", "Path where model params and summaries are saved")

flags.DEFINE_boolean("download_dataset", False, "Download MNIST dataset and persist in <data_dir> path.")

FLAGS = flags.FLAGS
NUM_IMAGES = {
    'train': 55000,
    'validation': 5000,
}


def main(argv):

    if FLAGS.download_dataset:
        # Download MNIST dataset and persist as binary files
        io.create_binary_files(FLAGS.data_dir)
        sys.exit(0)

    tf.logging.set_verbosity(tf.logging.INFO)

    classifier = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=FLAGS.model_dir,
        params={}
    )

    steps_per_epoch = NUM_IMAGES['train'] // FLAGS.batch_size
    print("Steps per epoch = {}".format(steps_per_epoch))

    for epoch in range(FLAGS.epochs):

        # Train
        training_input_fn, dataset_iterator_initializer_hook = preprocessing.get_training_inputs(FLAGS.batch_size,
                                                                                                 FLAGS.data_dir)
        classifier.train(input_fn=training_input_fn,
                             hooks=[dataset_iterator_initializer_hook],
                             steps=steps_per_epoch)

        # Evaluate
        metrics = classifier.evaluate(input_fn=preprocessing.get_test_inputs(FLAGS.batch_size, FLAGS.data_dir))

        print("Evaluation metrics: {}".format(str(metrics)))


def model_fn(features, labels, mode, params, config):
    """Sets up the graph for the model. Refer https://www.tensorflow.org/get_started/custom_estimators for details
    on the input parameters.
    """

    X = features["X"]
    y = labels
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    # Hyper-parameters
    layer1_neurons = params.get("layer1_neurons", 300)
    layer2_neurons = params.get("layer2_neurons", 100)
    dropout_keep_prob = 1 - params.get("dropout_rate", 0.2)
    grad_desc_learning_rate = params.get("grad_desc_learning_rate", 0.01)

    batch_norm_params = {'is_training': is_training, 'decay': 0.9}

    # DNN with two hidden layers, with dropout in-between
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):

        net = slim.fully_connected(X, layer1_neurons, scope='hl1')
        net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope="dropout")
        net = slim.fully_connected(net, layer2_neurons, scope='hl2')

        logits = slim.fully_connected(net, 10, activation_fn=None, scope='logits')

    # Loss
    with tf.name_scope('loss_op'):
        xentropy_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y)

    # Training op
    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=grad_desc_learning_rate)

        train_op = slim.learning.create_train_op(xentropy_loss, optimizer,
                                                 global_step=tf.train.get_or_create_global_step())

    # Accuracy metric
    with tf.name_scope("accuracy_op"):
        accuracy = tf.metrics.accuracy(y, tf.argmax(input=logits, axis=1))

    if mode == tf.estimator.ModeKeys.TRAIN:

        tf.summary.scalar('accuracy', accuracy[1])
        logging_hook = tf.train.LoggingTensorHook({"loss": xentropy_loss, "accuracy": accuracy[1]},
                                                  every_n_iter=100)

        return tf.estimator.EstimatorSpec(mode,
                                          loss=xentropy_loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook])

    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode,
                                          loss=xentropy_loss,
                                          eval_metric_ops={"accuracy": accuracy})

    else:   # mode == tf.estimator.ModeKeys.PREDICT
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=tf.argmax(input=logits, axis=1))


if __name__ == '__main__':
    tf.app.run()
