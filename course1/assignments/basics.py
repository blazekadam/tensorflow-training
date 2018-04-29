import numpy as np
import tensorflow as tf


def find_tensor_by_name(name: str) -> tf.Tensor:
    """
    Find and return the op with the given name in the default graph.

    :param name: op name
    :return: op with the given name from the default graph
    """
    return tf.get_default_graph().get_tensor_by_name(name+':0')


def populate_collection() -> None:
    """
    Create a custom collection named 'MY_COLLECTION' in the default graph and put a variable named 'coolsie'
    (shape (10, 10), dtype int32) to it.
    """
    graph = tf.get_default_graph()
    var = tf.get_variable('coolsie', shape=(10, 10), dtype=tf.int32)
    graph.add_to_collection('MY_COLLECTION', var)


def create_variable() -> tf.Variable:
    """
    Create and return variable name 'var' of type float32 and shape [100, 200] initialized to ones.
    """
    return tf.get_variable('var', shape=(100, 200), dtype=tf.float32, initializer=tf.ones_initializer())


def update_variable(var: tf.Variable) -> None:
    """
    Update the value of the given variable (returned from the function above) to zeros.

    :param var: variable of type float32 and shape [100, 200]
    """
    var.assign(np.zeros((100, 200))).eval()
