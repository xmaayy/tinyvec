import numpy as np

from tinyvec import VectorDB

import numpy as np
import pytest

from tinyvec import VectorDB

@pytest.fixture
def file_path(tmp_path):
    return tmp_path / "test_data.bin"

@pytest.fixture
def vdb_instance(file_path):
    return VectorDB(str(file_path), search_dim=4)


# Test the add method
def test_add(vdb_instance):
    vector = [1.0, 2.0, 3.0, 4.0]
    index = vdb_instance.add(vector)
    assert index == 0  # First vector should have index 0

# Test adding multiple vectors
def test_add_many(vdb_instance):
    vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    indices = vdb_instance.add_many(vectors)
    assert indices == [0, 1]  # Indices should be [0, 1] for the added vectors

# Test deleting a vector
def test_delete(vdb_instance):
    vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    indices = vdb_instance.add_many(vectors)
    moved_row, new_loc = vdb_instance.delete(0)
    assert moved_row == 1
    assert new_loc == 0

# Test getting k similar vectors
def test_get_k_similar_vecs(vdb_instance):
    vectors = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
    vdb_instance.add_many(vectors)
    query_vector = [1.0, 2.0, 3.0, 4.0]
    k = 2
    similar_indices = vdb_instance.get_k_similar_vecs(query_vector, k)
    assert set(similar_indices) == set([0, 1])  # Indices of the most similar vectors should be [0, 1]
