import numpy as np
import tensorflow as tf

from course1.assignments.mnist import build_mnist_optimizer, build_mnist_feed_forward, train_mnist
from course1.assignments.basics import find_tensor_by_name


def test_mnist_ff():
    with tf.Graph().as_default():
        with tf.Session().as_default():
            x = tf.placeholder(tf.float32, [None, 784], name='x')
            y_pred = build_mnist_feed_forward(x)
            assert y_pred.name == 'y_pred:0'
            assert y_pred.get_shape().as_list() == [None, 10]
            assert len(tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) > 0


def test_mnist_optimizer():
    with tf.Graph().as_default():
        with tf.Session().as_default():
            x = tf.placeholder(tf.float32, [None, 784], name='x')
            y = tf.placeholder(tf.float32, [None, 10], name='y')
            y_pred = build_mnist_feed_forward(x)
            train_op = build_mnist_optimizer(y, y_pred)
            assert type(train_op) == tf.Operation


def test_mnist_training(mnist_train_iterator, mnist_test_data):

    with tf.Graph().as_default():
        with tf.Session().as_default() as sess:
            train_mnist(mnist_train_iterator)

            x = find_tensor_by_name('x')
            y = find_tensor_by_name('y')
            y_pred = find_tensor_by_name('y_pred')

            valid_x, valid_y = mnist_test_data
            valid_y_pred = sess.run(y_pred, feed_dict={x: valid_x, y: valid_y})
            acc = np.mean(np.argmax(valid_y, 1) == np.argmax(valid_y_pred, 1))
            assert acc > 0.8
