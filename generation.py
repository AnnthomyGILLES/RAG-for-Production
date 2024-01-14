import os
from dotenv import load_dotenv
from openai import OpenAI

from chromadb.utils import embedding_functions
from config import config
from retrieval import augment_query_generated
from storage import StoreResults
from tools import word_wrap

load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API"))


class ResearchAssistant:
    def __init__(self):
        self.embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=config["models"]["embedding"]
            )
        )
        self.model = config["models"]["retrieval"]
        self.store = StoreResults()

    def augment_query(self, query):
        return augment_query_generated(query)

    def retrieve_documents(self, query):
        results = self.store.collection.query(
            query_texts=[query], n_results=5, include=["documents", "embeddings"]
        )
        return results["documents"][0]

    def generate_response(self, query, retrieved_documents):
        information = "\n\n".join(retrieved_documents)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful expert research assistant in artificial intelligence. Your users are asking questions about information contained in a documentation."
                "You will be shown the user's question, and the relevant information from documentation. Answer the user's question using only this information.",
            },
            {
                "role": "user",
                "content": f"Question: {query}. \n Information: {information}",
            },
        ]
        response = openai_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content

    def process_query(self, original_query):
        joint_query = self.augment_query(original_query)
        retrieved_documents = self.retrieve_documents(joint_query)
        output = self.generate_response(original_query, retrieved_documents)
        return word_wrap(output)


if __name__ == "__main__":
    assistant = ResearchAssistant()
    query = "what are the strategies to mitigate hallucination in llm"
    result = assistant.process_query(query)
    print(result)
