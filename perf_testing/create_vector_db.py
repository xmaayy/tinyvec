from pathlib import Path
from time import time, sleep
from tqdm import tqdm
import cProfile

from tinyvec import VectorDB
from sift_reader import ivecs_read, fvecs_read

if __name__ == "__main__":
    base_vecs = fvecs_read(Path(__file__).parent / "data/sift_base.fvecs")
    query_vecs = fvecs_read(Path(__file__).parent / "data/sift_query.fvecs")
    gt_vecs = ivecs_read(Path(__file__).parent / "data/sift_groundtruth.ivecs")
    sleep(5)
    # Construct the DB with base vectors
    vdb = VectorDB(search_dim=128, preallocate=2000000)
    sleep(5)
    # We dont really care about the indicies
    _ = vdb.add_many(base_vecs)