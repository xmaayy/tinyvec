from typing import List, Union

import numpy as np

VecLike = Union[np.ndarray, List[float]]


class LocalizationMixin:
    """
    A set of methods for reducing the search space for our Vector Database
    """

    @staticmethod
    def brute_force(vector: VecLike, vec_arr: np.ndarray):
        """
        Brute force means we dont want to localize so we just return
        all of the vectors for searching, not ideal for most cases but
        since we dont have to
        """
        return vec_arr
