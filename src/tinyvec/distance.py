from typing import Callable, Tuple

import numpy as np
import os

# Setting it to 1MB for chunk loading
MAX_MEMORY: int = int(os.environ.get("MAX_MEM_MB", 1)) * 1048576


class DistanceMixin:
    """
    All the very hot methods of calculating distance between two nDim
    points / vectors.
    """

    @staticmethod
    def euclidean_dist(x: np.ndarray, y: np.ndarray):
        diff = x - y
        return np.sqrt(np.dot(diff, diff))

    @staticmethod
    def euclidean_dist_square(x: np.ndarray, y: np.ndarray):
        diff = x - y
        return np.dot(diff, diff)

    @staticmethod
    def euclidean_dist_centred(x: np.ndarray, y: np.ndarray):
        diff = np.mean(x) - np.mean(y)
        return np.dot(diff, diff)

    @staticmethod
    def l1norm_dist(x: np.ndarray, y: np.ndarray):
        return sum(abs(x - y))

    @staticmethod
    def cosine_dist(x: np.ndarray, y: np.ndarray):
        return 1 - float(np.dot(x, y)) / ((np.dot(x, x) * np.dot(y, y)) ** 0.5)

    @staticmethod
    def _calculate_scores(
        vector: np.ndarray,
        locale: np.ndarray,
        scoring_fn: Callable,
        indices: Tuple[int, int] = None,
    ):
        if indices:
            return [
                scoring_fn(vector, dbvec)
                for idx, dbvec in enumerate(locale[indices[0] : indices[1] :])
            ]
        else:
            return [scoring_fn(vector, dbvec) for idx, dbvec in enumerate(locale[::])]

    @staticmethod
    def matrix_scoring_fn(target: np.ndarray, mat: np.ndarray):
        return np.linalg.norm(mat - target, axis=1)

    @staticmethod
    def _calculate_scores_multi(
        vector: np.ndarray,
        indices: Tuple[int, int],
        emb_dim: int,
        db_path: os.PathLike,
    ):
        row_bytes = emb_dim * np.dtype(np.float32).itemsize
        vec_arr = np.memmap(
            db_path,
            dtype=np.float32,
            mode="r",
            # Load only the data we need at this moment, this reduces
            # the reading and seeking for data massively
            shape=(max(indices) - min(indices), emb_dim),
            # We need to multiply the row index, times the number of elements in
            # each row to get the number of items we have to skip over, and then
            # multiply by the number of bytes per item to get the number of bytes
            # to skip over
            offset=min(indices) * row_bytes,
        )
        # The block size will generally be 4 or 8kb which means
        # as long as we're loading in blocks of 8 rows we should be
        # read efficient
        blocksize = os.statvfs(db_path).f_frsize
        min_rows = blocksize / (emb_dim * np.dtype(np.float32).itemsize)
        # Have to make sure we're not exceeding the max memory specified
        # as well
        chunk_size = int(MAX_MEMORY / (row_bytes * min_rows))
        scores = np.empty(vec_arr.shape[0])
        for i in range(0, vec_arr.shape[0], chunk_size):
            end_idx = min(i + chunk_size, vec_arr.shape[0])
            scores[i:end_idx] = DistanceMixin.matrix_scoring_fn(
                vector, vec_arr[i:end_idx, :]
            )
        return scores
