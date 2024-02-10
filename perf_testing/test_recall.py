from pathlib import Path
from time import time, sleep
import cProfile
import pstats

import numpy as np
import argparse

# Assuming VectorDB is updated with the necessary methods
from tinyvec import VectorDB  # Update the import according to your file structure

### Helpers - These remain unchanged
def bvecs_read(fname):
    a = np.fromfile(fname, dtype=np.int32, count=1)
    b = np.fromfile(fname, dtype=np.uint8)
    d = a[0]
    return b.reshape(-1, d + 4)[:, 4:].copy()

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--create", action="store_true", help="Create a new VectorDB")
    args = parser.parse_args()

    if args.create:
        base_vecs = fvecs_read(Path(__file__).parent / "data/sift/sift_base.fvecs")
        # Construct the DB with base vectors
        vdb = VectorDB(data_file_path="vector_db.bin", search_dim=128)
        for vec in base_vecs:
            vdb.add(vec.tolist())  # Convert numpy arrays to lists before adding
    else:
        vdb = VectorDB(data_file_path="vector_db.bin", search_dim=128)

    query_vecs = fvecs_read(Path(__file__).parent / "data/sift/sift_query.fvecs")
    gt_vecs = ivecs_read(Path(__file__).parent / "data/sift/sift_groundtruth.ivecs")

    for i in range(5):
        print(f"Starting recall test in {5-i}", end="\r")
        sleep(1)
        
    for k in range(2, 6):
        result_array = []
        for i, vec in enumerate(query_vecs[:10]):
            start = time()
            profiler = cProfile.Profile()
            profiler.enable()
            # The line to be profiled
            top_k_indices = vdb.get_k_similar_vecs(vec.tolist(), k, 1000)  # Convert numpy array to list
            profiler.disable()
            elapsed = time() - start
            # Create a Stats object based on the profiler's data and sort it by 'time' (total time in the function)
            stats = pstats.Stats(profiler).sort_stats('time')
            # Save the stats to a file
            stats_file = f'profiling_stats_k_{k}.txt'  # Filename example, adjust as needed
            with open(stats_file, 'w') as f:
                stats.stream = f
                stats.print_stats()
            # Convert ground truth indices for comparison
            gt_indices = set(gt_vecs[i, :k].flatten())
            # Calculate the difference between ground truth and obtained indices
            diff = gt_indices - set(top_k_indices)
            result_array.append((k - len(diff), elapsed))
        recall, times = zip(*result_array)
        avg_recall = sum(recall) / len(recall) / k
        avg_time = int(1000 * sum(times) / len(times))
        print(f"Avg Recall@{k}: {avg_recall:.4f} | "
            f"Avg Time: {avg_time}ms | "
            f"Base Size: {vdb.file_handler.rows}"
            )
    