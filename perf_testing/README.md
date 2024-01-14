# Performance Testing
What good is building something if you dont know how well it works? This section if the repository is dedicated to evaluating just how good (or bad) this toy vector database is at returning the data that we want to see.

# Datasets
## SIFT
[SIFT](http://corpus-texmex.irisa.fr/) is a collection of vector sets of different sizes that we can use to evaluate the distance metrics. The dimensionality of this dataset is already somewhat small when thinking in terms of natural language embeddings so it can be used without requiring PCA or any other dimensionality reduction.

## GIST
[GIST](http://corpus-texmex.irisa.fr/) is a single collection with the same structure as SIFT but with a dimension of almost 1000. This makes it a prime candidate for dimensionality reduction.

## Other Candidates
[COHERE](https://huggingface.co/datasets/Cohere/wikipedia-22-12) - A huge dataset of text from wikipedia in different languages that has already been embedded so we can use it directly in testing.

[C4](https://huggingface.co/datasets/allenai/c4) - Another massive (>12TB) of crawled data from the web.