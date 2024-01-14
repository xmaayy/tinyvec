from pathlib import Path
from time import time
from tqdm import tqdm

from tinyvec import VectorDB
from sift_reader import ivecs_read, fvecs_read

if __name__ == "__main__":
    base_vecs = fvecs_read(Path(__file__).parent / "data/siftsmall_base.fvecs")
    query_vecs = fvecs_read(Path(__file__).parent / "data/siftsmall_query.fvecs")
    gt_vecs = ivecs_read(Path(__file__).parent / "data/siftsmall_groundtruth.ivecs")

    # Construct the DB with base vectors
    vdb = VectorDB(search_dim=128, preallocate=10001)
    # We dont really care about the indicies
    _ = vdb.add_many(base_vecs)
    
    for k in range(1,6):
        result_array = []
        for i, vec in enumerate(query_vecs[::]):
            start = time()
            _, closest = vdb.get_k_similar_vecs(vec, k)
            elapsed = time() - start
            # They're unsorted so its probably fastest for us to just compare
            # them as a set
            diff = set(closest) - set(gt_vecs[i, :5])
            result_array.append((5-len(diff), elapsed))
        recall, times = zip(*result_array)
        avg_recall = (sum(recall)/len(recall))/5
        avg_time = int(1000*sum(times)/len(times))
        print(f"Avg Recall@{k} {avg_recall} | "
              f"Avg Time {avg_time}ms | "
              f"Base Size {base_vecs.shape[0]}"
            )
        

