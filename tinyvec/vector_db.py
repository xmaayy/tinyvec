from typing import Optional, Union, List, Tuple
import numpy as np
import shutil
import math
import os

from .localization import LocalizationMixin
from .distance import DistanceMixin

VecLike = Union[np.ndarray, List[float]]


class VectorDB(LocalizationMixin, DistanceMixin):
    """
    An on-disk implementation of a vector database. As with anything made in the
    spirit of learning there are tradeoffs, and this specific implementations
    tradeoff is that it is build to be very infrequently written to but read more
    frequently. This is thanks to numpy memory mapped arrays not having any mechanism
    for in-place resizing, so the entire binary file has to be re-created whenever
    it runs out of space. We can try to minimize this by pre-allocating in chunks
    but there will still be a non-insignificant delay when moving between chunks.
    """

    data_file_path: os.PathLike
    """The path to the file that we are storing the vectors in."""

    search_dim: int
    """The size that we actually want to store vectors as after they are ingested
    # by the database."""

    vec_arr: np.ndarray
    """Memory mapped array of vectors stored by numpy to hold the vectors"""

    end_index: int
    """Space Counter | Having to pre-allocate array chunks means that we will need
    to keep track of how many spaces we have left to allocate. Numpy memmap arrays
    are initialized as 0's, so when we load a memmap array we can count backwards
    from the end to determine how many remaining spots there are"""

    reallocation_fraction: float
    """Reallocation fraction (0-1] | You can balance the unused space that gets pre-allocated
    with the frequency at which you need to re-allocate by tuning this parameter. If you
    set it very low, you will be more frequently reallocating space. If you set it
    high, you will less frequently allocate space but you will end up with more unused space
    (Set it high for short lived experiments, low for longer term data)"""

    def __init__(
        self,
        data_file_path: Optional[os.PathLike] = "VecDB.bin",
        search_dim: int = 64,
        preallocate: int = 1000,
        reallocation_fraction: float = 0.50,
    ):
        self.data_file_path = data_file_path
        self.vec_arr: np.ndarray = np.memmap(
            data_file_path, dtype="float32", mode="w+", shape=(preallocate, search_dim)
        )

        self.reallocation_fraction = reallocation_fraction
        self.end_index = 0
        self.search_dim = search_dim

    def __reallocate(self, n: Optional[int] = None):
        """Reallocate the array to be larger. This is a very expensive operation
        because it requires copying the entire array to a new location. This is
        why we want to pre-allocate as much space as we can to avoid this operation

        Args:
            n (Optional[int], optional): New vectors needed. Defaults to None.
        """
        new_size = self.vec_arr.shape[0] + math.ceil(
            self.vec_arr.shape[0] * self.reallocation_fraction
        )
        # If we have a goal size, run the reallocation calculation until
        # the new size is greater than the required size
        # TODO: This can currently over-allocate by one iteration
        # because it does not account for the currently available slots
        while n and (new_size < (self.vec_arr.shape[0] + n)):
            new_size = new_size + math.ceil(new_size * self.reallocation_fraction)
        temp_path = str(self.data_file_path) + "1"
        new_array: np.ndarray = np.memmap(
            temp_path,
            dtype="float32",
            mode="w+",
            shape=(new_size, self.vec_arr.shape[1]),
        )
        # TODO: Maybe make the chunk size configurable in the future to
        # lessen the pain of a lower reallocation_fraction
        chunk_size = 1024
        for i in range(math.ceil(self.vec_arr.shape[0] / chunk_size)):
            begin = i * chunk_size
            end = min((i + 1) * chunk_size, self.vec_arr.shape[0])
            new_array[begin:end, :] = self.vec_arr[begin:end, :]
            new_array.flush()
        # Want to explicitly close these files because the interaction between
        # mmaps and file ref counters is weird.
        new_array._mmap.close()
        self.vec_arr._mmap.close()

        shutil.move(temp_path, self.data_file_path)
        self.vec_arr = np.memmap(
            self.data_file_path,
            dtype="float32",
            mode="r+",
            shape=(new_size, self.search_dim),
        )

    def add(self, vector: VecLike):
        """
        Add a new vector to the database. This method is inefficient and should
        only be used when adding single vectors because it has to resize the array
        every time it is called.
        """
        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32, ndmin=2)
        # if the end index is at the end of the current array chunk, we should
        # pre-allocate another chunk of data whose size is determined by the
        # reallocation fraction.
        if self.end_index >= self.vec_arr.shape[0]:
            self.__reallocate()
        self.vec_arr[self.end_index, :] = vector
        self.end_index += 1
        # Return the index of the added string
        return self.end_index - 1

    def add_many(self, vectors: Union[np.ndarray, List[List[float]]]):
        """This is the more efficient way to add vectors to the DB, particularily
        when considering the cost of reallocation as this function will try to
        only call the reallocation/copy function once for the whole block
        """
        if isinstance(vectors, list):
            vectors = np.array(vectors, dtype=np.float32, ndmin=2)
        num_vecs = vectors.shape[0]
        # If adding this chunk would put us over the size we have available
        # then we should allocate enough space to accomodate the chunk
        if (self.end_index + num_vecs) >= self.vec_arr.shape[0]:
            self.__reallocate(num_vecs)
        self.vec_arr[self.end_index : (self.end_index + num_vecs), :] = vectors
        self.end_index += num_vecs
        # Return the index of the added string
        return np.array(range(self.end_index - num_vecs, self.end_index))

    def _get_group(self, vector: VecLike, method: str):
        if hasattr(self, method):
            localization_method = getattr(self, method)
        else:
            raise NotImplementedError(f"No localization method | {method}")
        locale: np.ndarray = localization_method(vector, self.vec_arr)

        return locale

    def get_k_similar_vecs(
        self,
        vector: VecLike,
        k: int = 5,
        metric: str = "euclidean_dist_square",
        localization: str = "brute_force",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the k most similar vectors to the given vector

        Args:
            vector (VecLike): The vector to compare against
            k (int, optional): Number of vectors to return. Defaults to 5.
            metric (str, optional): Distance metric to use. Defaults to "euclidean_dist_square".
            localization (str, optional): Localization method to use. Defaults to "brute_force".

        Raises:
            NotImplementedError: If the metric or localization method is not implemented

        Returns:
            Tuple[np.ndarray, np.ndarray]:
        """
        locale = self._get_group(vector, localization)
        if hasattr(self, metric):
            scoring_fn = getattr(self, metric)
        else:
            raise NotImplementedError(f"No similarity metric | {metric}")
        scores = [scoring_fn(vector, dbvec) for idx, dbvec in enumerate(locale[::])]
        sim_vec = np.argpartition(scores, k)[:k]
        # Returned in no particular order
        return locale[sim_vec, :], sim_vec

    @staticmethod
    def _pca(dset, topk):
        """
        Numpy only PCA for dimensionality reduction
        """
        means = np.mean(dset, axis=0)
        stddevs = np.std(dset, axis=0, ddof=1)
        stdset = (dset - means) / stddevs
        # Next we need to calculate the covariance matrix
        covmat = np.dot(stdset.T, stdset) / (4 / 0.8)
        # Get eigen vals/vecs
        evals, evecs = np.linalg.eig(covmat)
        # Choose top-k PCS based on eigenvals
        # argpartition is linear time worst case
        pt = np.argpartition(evals, -topk)[-topk:]
        # Dot the eigen vec and the feature vec together
        # to get the principal component vectors
        return np.dot(stdset, evecs[:, pt])
