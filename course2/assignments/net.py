import tensorflow as tf


def build_facial_keypoints_ff(images: tf.Tensor) -> tf.Tensor:
    """
    Build CNN regression feed-forward network from the given image to key-point coordinates.
    - images: [None, 96, 96, 1] float32
    - outputs: [None, 8] float32

    Tips:
    - scale images to 0-1 (from 0-255)
    - use tf.layers.conv2d and tf.layers.max_pooling2d and final tf.layers.dense layer
    - do not forget to select activation function
    - 32-64 channels should provide satisfactory results

    :param images: input images tensor
    :return: output predictions tensor
    """
