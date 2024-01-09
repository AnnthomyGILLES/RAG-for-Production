import os
from typing import Any, Dict, List

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from loguru import logger

import chromadb

load_dotenv()

EMBEDDING_MODEL = "text-embedding-ada-002"

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API"), model_name=EMBEDDING_MODEL
)


class StoreResults:
    """
    Singleton class for storing results in a ChromaDB collection.

    This class is responsible for initializing and interacting with a ChromaDB client,
    and provides functionality to add batches of documents to the ChromaDB collection.
    It uses a singleton pattern to ensure only one instance manages the database connection.

    Methods:
        __call__(self, batch: Dict[str, Any]) -> Dict: Processes and adds a batch of documents to the collection.
    """

    _instance = None

    def __init__(self):
        self.collection = self.chroma_client.get_or_create_collection(
            name="ray-documentation",
            embedding_function=embedding_function,
        )

    def __new__(cls):
        # Create instance if not already created
        if cls._instance is None:
            cls._instance = super(StoreResults, cls).__new__(cls)
            # Initialize ChromaDB client
            cls._instance.chroma_client = chromadb.PersistentClient(path="./chromadb")
        return cls._instance

    def __call__(self, batch: Dict[str, Any]) -> Dict:
        """
        Processes and adds a batch of documents to the collection.

        Args:
            batch (Dict[str, Any]): A batch of documents to be stored.
                                    Expected keys are 'text', 'embeddings', 'index', and 'source'.

        Returns:
            Dict: An empty dictionary, potentially can be modified to return status or results.
        """
        try:
            documents: List[str] = batch["text"].tolist()
            embeddings: List[Any] = batch["embeddings"].tolist()
            ids: List[int] = batch["index"].tolist()
            metadatas: List[Dict[str, str]] = [{"source": value} for value in batch["source"]]

            self.collection.add(
                embeddings=embeddings, documents=documents, metadatas=metadatas, ids=ids
            )
            logger.info(f"Successfully upserted {len(documents)} documents.")
        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
        return {}


if __name__ == "__main__":
    store = StoreResults()
    results = store.collection.query(
        query_texts=["Tell me about Head Node"], n_results=2
    )
    print(results)
