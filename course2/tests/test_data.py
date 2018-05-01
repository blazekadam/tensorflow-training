import numpy as np

from course2.assignments.data import batch_data


def test_load_data(data):
    images, targets = data
    assert images.shape == (8, 96, 96, 1)
    assert images.dtype == np.uint8
    assert targets.shape == (8, 4, 2)
    assert targets.dtype == np.float32


def test_batch_data(data):
    batch_images, batch_targets = next(batch_data(*data))  # batch size 100
    assert batch_images.shape == (8, 96, 96, 1)
    assert batch_targets.shape == (8, 4, 2)

    assert len(list(batch_data(*data, batch_size=2))) == 4
    for batch_images, batch_targets in batch_data(*data, batch_size=2):
        assert batch_images.shape == (2, 96, 96, 1)
        assert batch_targets.shape == (2, 4, 2)

    # test shuffling
    some_images, _ = next(batch_data(*data, batch_size=4))
    other_images, _ = next(batch_data(*data, batch_size=4))
    assert np.not_equal(some_images, other_images).any()
