import tensorflow as tf
import pytest
import numpy as np


from course1.assignments.basics import find_tensor_by_name, populate_collection, create_variable, update_variable


@pytest.mark.parametrize('name', ('my_name', 'another_name'))
def test_find_tensor(name):

    with tf.Graph().as_default():
        with tf.Session().as_default():
            tf.get_variable(name, shape=(10, 3), dtype=tf.int32)
            tensor = find_tensor_by_name(name)
            assert tensor.name == name+':0'

    with tf.Graph().as_default():
        with tf.Session().as_default():
            tf.placeholder(tf.int32, shape=(10, 3), name=name)
            tensor = find_tensor_by_name(name)
            assert tensor.name == name+':0'

    with tf.Graph().as_default():
        with tf.Session().as_default():
            input_ = tf.placeholder(tf.int32, shape=(10, 3), name='input')
            a = tf.reduce_sum(input_, name=name)
            tensor = find_tensor_by_name(name)
            assert a == tensor


def test_collection():
    with tf.Graph().as_default():
        coll = tf.get_default_graph().get_collection('MY_COLLECTION')
        assert len(coll) == 0

        populate_collection()

        coll = tf.get_default_graph().get_collection('MY_COLLECTION')
        assert len(coll) == 1
        var = coll[0]
        assert tf.identity(var).dtype == tf.int32
        assert var.get_shape().as_list() == [10, 10]
        assert var.name == 'coolsie:0'


def test_variable():

    with tf.Graph().as_default():
        with tf.Session().as_default():
            var = create_variable()
            tf.global_variables_initializer().run()
            assert var.get_shape().as_list() == [100, 200]
            assert tf.identity(var).dtype == tf.float32
            np.testing.assert_allclose(var.eval(), np.ones((100, 200)))
            update_variable(var)
            np.testing.assert_allclose(var.eval(), np.zeros((100, 200)))
