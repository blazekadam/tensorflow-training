import tensorflow as tf


def find_tensor_by_name(name: str) -> tf.Tensor:
    """
    Find and return the op with the given name in the default graph.

    :param name: op name
    :return: op with the given name from the default graph
    """


def populate_collection() -> None:
    """
    Create a custom collection named 'MY_COLLECTION' in the default graph and put a variable named 'coolsie'
    (shape (10, 10), dtype int32) to it.
    """
