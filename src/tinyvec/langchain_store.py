"""A LangChain compatible VectorStore wrapper for TinyVec"""
from typing import Iterable, List, Optional, Any, Tuple, Callable, Dict, TypeVar
from time import time

from langchain_core.vectorstores import VectorStore
from langchain_core.runnables.config import run_in_executor
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from .vector_db import VectorDB

VST = TypeVar("VST", bound="VectorStore")


class LangchainVectorDB(VectorStore):
    """A LangChain compatible VectorStore wrapper for TinyVec"""

    embedder: Embeddings
    """The class responsible for converting text to list of floats"""

    texts: Dict[int, str]
    """Holds a relation between an index into the VectorDB and a string of text"""

    vdb: VectorDB
    """Holds the vectors and does the relevance search"""

    default_k: int
    """The default number of similar examples to return when querying for documents"""

    debug_logger: Callable
    """How to log debug messages by default"""

    def __init__(
        self,
        vdb: VectorDB,
        embedder: Embeddings,
        texts: Dict[int, str] = {},
        default_k: int = 4,
        debug_logger: Callable = lambda _: None,
        **kwargs,
    ) -> "LangchainVectorDB":
        super().__init__(**kwargs)  # Fixed to call super().__init__ with kwargs
        self.vdb = vdb
        self.embedder = embedder
        self.texts = texts
        self.default_k = default_k
        self.debug_logger = debug_logger

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedder: Embeddings,
        emb_dim: int = None,
        individually: bool = False,
        default_k: int = 4,
        debug_logger: Callable = lambda x: None,
        **kwargs,
    ):
        """Return VectorStore initialized from texts and embeddings."""
        vdb = VectorDB(search_dim=emb_dim)
        inds = []
        if individually:
            debug_logger("Embedding individual queries")
            for query in texts:
                s = time()
                embd = embedder.embed_query(query)
                debug_logger(f"Time to embed {time() - s}")
                ind = vdb.add(embd)  # Updated to use the return value of add
                inds.append(ind)
        else:
            debug_logger("Embedding all docs at once")
            embds = embedder.embed_documents(texts)
            inds = vdb.add_many(embds)  # Updated to use the return value of add_many
        debug_logger("Done")
        lcvdb = cls(
            vdb, embedder, dict(zip(inds, texts)), default_k, debug_logger, **kwargs
        )
        return lcvdb

    def add_texts(
        self,
        texts: Iterable[str],
        **kwargs,
    ) -> List[int]:
        """Embed and add texts to the VectorDB, returning their indices."""
        embedded_data = self.embedder.embed_documents(texts)
        indices = self.vdb.add_many(embedded_data)  # Use add_many's returned indices
        new_elements = dict(zip(indices, texts))
        self.texts.update(new_elements)  # Use update for merging dictionaries
        return indices

    def delete(self, ids: List[int], **kwargs) -> None:
        """Delete vectors by their indices."""
        for id in ids:
            moved_index, new_location = self.vdb.delete_row(id)
            # Handle updates to self.texts if necessary, considering moved vectors
            if moved_index is not None:
                self.texts[new_location] = self.texts.pop(moved_index)

    @classmethod
    def from_documents(
        cls: type[VST], documents: List[Document], embedding: Embeddings, **kwargs: Any
    ) -> VST:
        return cls.from_texts([d.content for d in documents], embedding, kwargs)

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return None if not hasattr(self, "embedder") else self.embedder

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        return await run_in_executor(None, self.delete, ids, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        vector = self.embedder.embed_query(query)
        _, inds = self.vdb.get_k_similar_vecs(
            vector, k, "euclidean_dist_square", "brute_force"
        )
        return [self.texts[i] for i in inds]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.

        Vectorstores should define their own selection based method of relevance.
        """
        raise NotImplementedError

    def similarity_search_with_score(
        self, *args: Any, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        raise NotImplementedError

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        raise NotImplementedError

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        raise NotImplementedError

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance."""
        raise NotImplementedError
