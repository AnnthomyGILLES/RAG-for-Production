import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv()

OPENAI_API = os.getenv("OPENAI_API")


class EmbedChunks:
    """
    A class to handle embedding of text chunks using either OpenAI or HuggingFace models.

    This class initializes an embedding model based on the provided model name.
    If the model name matches "text-embedding-ada-002", it uses OpenAI's embeddings model.
    For any other model names, it defaults to using HuggingFace's embedding models.

    Attributes:
        embedding_model (OpenAIEmbeddings or HuggingFaceEmbeddings): The embedding model instance.

    Args:
        model_name (str): The name of the model to be used for embeddings.
                          If "text-embedding-ada-002", OpenAI's model is used; otherwise, HuggingFace's model is used.
    """

    def __init__(self, model_name):
        """
        Initialize the EmbedChunks class with a specified embedding model.

        Args:
            model_name (str): The name of the model to be used for generating embeddings.
        """
        if model_name == "text-embedding-ada-002":
            self.embedding_model = OpenAIEmbeddings(
                model=model_name, openai_api_key=OPENAI_API
            )
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                # model_kwargs={"device": "cuda"},
                # encode_kwargs={"device": "cuda", "batch_size": 100},
            )

    def __call__(self, batch):
        """
        Compute embeddings for a batch of text documents.

        This method takes a batch of text documents, computes their embeddings using the initialized model,
        and returns the batch along with their corresponding embeddings.

        Args:
            batch (dict): A batch of text documents, structured as a dictionary with keys 'text' and 'source',
                          where 'text' is a list of document texts and 'source' is their respective sources.

        Returns:
            dict: A dictionary with keys 'text', 'source', and 'embeddings'. The 'embeddings' key contains
                  the computed embeddings corresponding to each text in the batch.
        """
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {
            "text": batch["text"],
            "source": batch["source"],
            "embeddings": embeddings,
        }
