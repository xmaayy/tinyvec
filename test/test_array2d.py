import pytest
import numpy as np
import os
from tinyvec.array2d import Array2DFileHandler  # Adjust the import as necessary
CONST_DIM = 10
@pytest.fixture
def file_path(tmp_path):
    return tmp_path / "test_data.bin"

@pytest.fixture
def single_precision_handler(file_path):
    return Array2DFileHandler(str(file_path), rows=0, cols=CONST_DIM, precision='single')

@pytest.fixture
def double_precision_handler(file_path):
    return Array2DFileHandler(str(file_path), rows=0, cols=CONST_DIM, precision='double')

def create_test_data(rows, cols, precision='single'):
    dtype = np.float32 if precision == 'single' else np.float64
    return np.random.rand(rows, cols).astype(dtype)

def read_raw_data(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
    return data

def test_file_initialization(single_precision_handler, file_path):
    print(file_path)
    assert file_path.exists()
    data = read_raw_data(file_path)
    assert len(data) == 12  # Header size

def test_write_and_read_single(single_precision_handler):
    test_data = create_test_data(10, CONST_DIM)
    for row in test_data:
        single_precision_handler.write_row(row)
    read_data = single_precision_handler.read()
    np.testing.assert_array_almost_equal(test_data, read_data)

def test_write_and_read_double(double_precision_handler):
    test_data = create_test_data(10, CONST_DIM, precision='double')
    double_precision_handler.write(test_data.tolist())
    read_data = double_precision_handler.read()
    np.testing.assert_array_almost_equal(test_data, read_data)

def test_delete_row(single_precision_handler):
    test_data = create_test_data(10, CONST_DIM)
    single_precision_handler.write(test_data.tolist())
    moved_index, new_location = single_precision_handler.delete_row(2)
    assert moved_index == 9  # The last row should move to the deleted row's position
    assert new_location == 2
    # Verify data integrity
    remaining_data = single_precision_handler.read()
    assert len(remaining_data) == 9 # One row less
    np.testing.assert_array_almost_equal(test_data[[0, 1, 9, 3, 4, 5, 6, 7, 8], :], remaining_data)

def test_lock_handling(single_precision_handler):
    # Attempt to write data with the lock set manually
    with open(single_precision_handler.filepath, 'r+b') as file:
        single_precision_handler._set_lock(file, True)
        with pytest.raises(Exception):
            single_precision_handler.write([[1.0]])
        single_precision_handler._set_lock(file, False)
