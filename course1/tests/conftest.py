import pytest
from tensorflow.examples.tutorials.mnist import input_data


@pytest.fixture()
def mnist_train_iterator():
    def mnist_iter():
        mnist = input_data.read_data_sets(str('.mnist_data'), one_hot=True)
        for i in range(1000):
            yield mnist.train.next_batch(100)
    yield iter(mnist_iter())


@pytest.fixture()
def mnist_test_data():
    mnist = input_data.read_data_sets(str('.mnist_data'), one_hot=True)
    yield mnist.validation.images, mnist.validation.labels
