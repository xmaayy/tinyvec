from typing import List, Tuple

import numpy as np

from .array2d import Array2DFileHandler

class VectorDB():
    file_handler: Array2DFileHandler
    """File handler for the vector data file"""

    def __init__(
        self, 
        data_file_path: str = "VecDB.bin", 
        search_dim: int = 64, 
    ):
        """
        Initialize a VectorDB object.

        Args:
            data_file_path (str, optional): Path to the vector data file. Defaults to "VecDB.bin".
            search_dim (int, optional): Dimension of the vectors. Defaults to 64.
        """
        self.data_file_path = data_file_path
        self.search_dim = search_dim
        
        # Initialize the file handler with 0 rows initially; it will grow dynamically as vectors are added.
        self.file_handler = Array2DFileHandler(data_file_path, 0, search_dim, 'single')
    
    def add(self, vector: list[float]) -> int:
        """
        Add a vector to the database.

        Args:
            vector (list[float]): The vector to be added.

        Returns:
            int: The index of the newly added vector.
        
        Raises:
            ValueError: If the size of the vector does not match the search dimension.
        """
        if len(vector) != self.search_dim:
            raise ValueError("Vector size does not match search dimension.")
        
        # The index of the new vector is the current number of rows
        new_index = self.file_handler.rows
        # Dynamically add the new vector as a new row at the end.
        self.file_handler.write_row(vector)
        return new_index
    
    def add_many(self, vectors: list[list[float]]) -> list[int]:
        """
        Add multiple vectors to the database.

        Args:
            vectors (list[list[float]]): The list of vectors to be added.

        Returns:
            list[int]: The indices of the newly added vectors.
        
        Raises:
            ValueError: If any vector in the list does not match the search dimension.
        """
        if any(len(vector) != self.search_dim for vector in vectors):
            raise ValueError("One or more vectors do not match the search dimension.")
        
        # The index of the first new vector is the current number of rows
        start_index = self.file_handler.rows
        indices = []
        for i, vector in enumerate(vectors):
            # Dynamically add each vector. Optimized to add all at once if possible.
            self.file_handler.write_row(vector)
            indices.append(start_index + i)
        return indices

    def delete(self, index: int) -> Tuple[int, int]:
        """
        Delete a vector from the database.

        Args:
            index (int): The index of the vector to be deleted.

        Returns:
            Tuple[int, int]: A tuple containing the index of the deleted vector and the number of remaining vectors.
        
        Raises:
            ValueError: If the index is out of range.
        """
        if index < 0 or index >= self.file_handler.rows:
            raise ValueError("Index out of range.")
        return self.file_handler.delete_row(index)
   
    def get_k_similar_vecs(
        self,
        vector: List[float],
        k: int = 5,
        batch_size: int = 100  # Define a suitable batch size for your environment
    ) -> List[int]:
        """
        Get the indices of the k most similar vectors to the given vector, utilizing batch processing.

        Args:
            vector (List[float]): The vector to compare against.
            k (int, optional): Number of vector indices to return. Defaults to 5.
            batch_size (int, optional): The number of vectors to process in each batch.

        Raises:
            NotImplementedError: If the metric or localization method is not implemented.

        Returns:
            List[int]: The indices of the top k similar vectors.
        """
        vector_np = np.array(vector)  # Convert the input vector to a NumPy array once

        # Initialize an empty list to store scores and vector indices
        all_scores = np.array([])
        all_indices = np.array([])

        total_rows = self.file_handler.rows
        for batch_start in range(0, total_rows, batch_size):
            batch = self.file_handler.load_batch(batch_start, batch_start + batch_size)
            # Convert batch to 2D array if necessary, for now assuming `batch` is already in the correct format
            batch_scores = np.linalg.norm(vector_np-batch, axis=1)
            # Append scores and their corresponding indices to the scores list
            all_scores = np.concatenate((all_scores, batch_scores))
            all_indices = np.concatenate((all_indices, np.arange(batch_start, batch_start + batch_scores.size)))

        # Use argpartition to find the indices of the top k scores efficiently
        top_k_indices = all_indices[np.argpartition(all_scores, k)[:k]].astype(int)

        return top_k_indices
