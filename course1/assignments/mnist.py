import tensorflow as tf

from typing import Iterable, Tuple


def build_mnist_feed_forward(x: tf.Tensor) -> tf.Tensor:
    """
    Build a MNIST feed-forward network with a single fully connected layer and RELU activation,
    return the output tensor (shaped [None, 10]).

    Use tf.layers or create your own variables.

    Name the output tensor as 'y_pred'.

    What variables initializer did you use?

    :param x: input images of shape [None, 784]
    :param y: target classes of shape [None, 10]
    :return: y predictions of shape [None, 10]
    """


def build_mnist_optimizer(y, y_pred) -> tf.Operation:
    """
    Build a trainable MNIST network and return the training operation.

    Use
    - tf.nn.softmax_cross_entropy_with_logits
    - tf.train.GradientDescentOptimizer

    Name the placeholders x and y.
    Name the output tensor as 'y_pred'.

    :param y: target classes of shape [None, 10]
    :param y_pred: target predictions of shape [None, 10]
    :return: training operation
    """


def train_mnist(batch_iterator: Iterable[Tuple[tf.Tensor, tf.Tensor]]) -> None:
    """
    Build a trainable MNIST network and run the training on the given iteration of batches.

    Use the builder functions above.

    Name the input tensors x and y.
    Name the output tensor y_pred.

    Use tf.get_default_session() for the training and do not forget to initialize the variables.

    Optional:
        - compute accuracy in tensorflow
        - create summary and FileWriter (https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
        - see the training progress in TensorBoard
        - add L2 variable regularization to REGULARIZATION_LOSSES collection
          and use this collection when creating the loss

    :param batch_iterator: MNIST training x, y tuples iterator ([None, 784], [None, 10])
    """
