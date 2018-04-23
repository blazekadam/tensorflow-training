import sys; sys.path.append('..')  # allow to import other stuff
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from course1.assignments.basics import find_tensor_by_name
from course1.assignments.mnist import train_mnist
from course1.tests.conftest import mnist_train_iterator, mnist_test_data

mnist = input_data.read_data_sets(str('.mnist_data'), one_hot=True)

with tf.Graph().as_default():
    with tf.Session().as_default() as sess:
        train_mnist(next(mnist_train_iterator()))

        x = find_tensor_by_name('x')
        y = find_tensor_by_name('y')
        y_pred = find_tensor_by_name('y_pred')

        valid_x, valid_y = next(mnist_test_data())
        valid_y_pred = sess.run(y_pred, feed_dict={x: valid_x, y: valid_y})
        acc = np.mean(np.argmax(valid_y, 1) == np.argmax(valid_y_pred, 1))
        print('Test accuracy: {}'.format(acc))
