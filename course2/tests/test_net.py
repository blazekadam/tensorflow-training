import tensorflow as tf

from course2.assignments.net import build_facial_keypoints_ff


def test_build_facial_keypoints_ff():
    images = tf.placeholder(tf.float32, shape=[None, 96, 96, 1], name='images')
    predictions = build_facial_keypoints_ff(images)
    assert predictions.get_shape().as_list() == [None, 8]
    assert predictions.dtype == tf.float32
    assert tf.get_default_graph().get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
