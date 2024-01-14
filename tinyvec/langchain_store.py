"""A LangChain compatible VectorStore wrapper for TinyVec"""
from typing import Iterable, List, Optional, Any, Type, Tuple, Callable, Dict, TypeVar
from abc import abstractmethod

from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.runnables.config import run_in_executor
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from .vector_db import VectorDB

VST = TypeVar("VST", bound="VectorStore")

class LangchainVectorDB(VectorStore):
    """A LangChain compatible VectorStore wrapper for TinyVec"""

    # The class responsible for converting text to list of floats
    embedder: Embeddings

    # Holds a relation between an index into the VectorDB and a string of text
    texts: Dict[int, str]

    # Does all the cool vector stuff
    vdb: VectorDB

    def add_texts(
        self,
        texts: Iterable[str],
        **kwargs: Any,
    ) -> List[str]:
        embedded_data = self.embeddings.embed_documents(texts)
        indices = self.add_many(embedded_data)
        new_elements = dict(zip(indices, texts))
        return indices

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return None if not hasattr(self, 'embedder') else self.embedder

    def delete(self, ids: List[str] = None, **kwargs: Any) -> bool:
        self.vdb.remove_many(ids)

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        return await run_in_executor(None, self.delete, ids, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Return docs most similar to query."""
        vector = self.embedder.embed_query(query)
        self.vdb.get_k_similar_vecs(vector, k, "euclid", "brute_force")

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

    @classmethod
    def from_texts(
        cls: 'LangchainVectorDB',
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        emb_dim: int = None,
        individually: bool = False,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from texts and embeddings."""
        lcvdb = LangchainVectorDB()
        vdb = VectorDB(search_dim=emb_dim, preallocate=len(texts))
        if individually:
            embds = embedding.embed_query(texts)
            inds = vdb.add(embds)
        else:
            embds = embedding.embed_documents(texts)
            inds = vdb.add_many(embds)
        lcvdb.vdb = vdb
        lcvdb.embedder = embedding
        lcvdb.texts = dict(zip(inds, texts))
        return lcvdb