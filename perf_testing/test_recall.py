from pathlib import Path
from time import time, sleep
from tqdm import tqdm
import cProfile

from tinyvec import VectorDB
from sift_reader import ivecs_read, fvecs_read

if __name__ == "__main__":
    query_vecs = fvecs_read(Path(__file__).parent / "data/sift_query.fvecs")
    gt_vecs = ivecs_read(Path(__file__).parent / "data/sift_groundtruth.ivecs")
    # Allowing time for memory to settle so we dont confuse it with
    # what we're using
    sleep(1)

    vdb = VectorDB("VecDB.bin", search_dim=128)

    sleep(1)
    
    for k in range(1,6):
        result_array = []
        for i, vec in enumerate(query_vecs[:20:]):
            start = time()
            _, closest = vdb.get_k_similar_vecs(vec, k, multiproc=False)
            elapsed = time() - start
            # They're unsorted so its probably fastest for us to just compare
            # them as a set
            diff = set(gt_vecs[i, :k]) - set(closest)
            result_array.append((k-len(diff), elapsed))
        recall, times = zip(*result_array)
        avg_recall = (sum(recall)/len(recall))/k
        avg_time = int(1000*sum(times)/len(times))
        print(f"Avg Recall@{k} {avg_recall} | "
              f"Avg Time {avg_time}ms | "
              f"Base Size {vdb.size}"
            )
        

