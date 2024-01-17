import numpy as np


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
