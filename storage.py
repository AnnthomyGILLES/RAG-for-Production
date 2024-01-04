import os

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from loguru import logger

import chromadb

load_dotenv()
# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-ada-002"

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API"), model_name=EMBEDDING_MODEL
)


class StoreResults:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chromadb")
        self.collection = self.chroma_client.get_or_create_collection(
            name="ray-documentation",
            embedding_function=embedding_function,
        )

    def __call__(self, batch):
        try:
            # Prepare the data for insertion
            embeddings, documents, metadatas, ids = [], [], [], []

            for text, source, embedding, index in zip(
                    batch["text"], batch["source"], batch["embeddings"], batch["index"]
            ):
                logger.info(f"{text} ----- {source} ------ .")
                embeddings.append(embedding)
                documents.append(text)
                metadatas.append({"source": source})
                ids.append(index)

            # Upsert the data into the collection
            # Assuming your collection's `add` method supports upserting
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
