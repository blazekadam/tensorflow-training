import os.path as path

import pytest

from course2.assignments.data import load_data


@pytest.fixture()
def data():
    yield load_data(path.join('tests', 'training_sample.csv'))
