import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

load_dotenv()

OPENAI_API = os.getenv("OPENAI_API")


class EmbedChunks:
    def __init__(self, model_name):
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
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {
            "text": batch["text"],
            "source": batch["source"],
            "embeddings": embeddings,
        }
