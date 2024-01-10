import os

from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores.chroma import Chroma
from openai import OpenAI

from embeddings import EmbedChunks
from storage import StoreResults

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API"))
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


def rag(query, retrieved_documents, model="gpt-3.5-turbo"):
    information = "\n\n".join(retrieved_documents)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert assistant. Your users are asking questions about information contained in a documentation."
                       "You will be shown the user's question, and the relevant information from documentation. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {information}"}
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    return content


if __name__ == '__main__':
    # query = "what are the strategies to mitigate hallucination in llm"
    # store = StoreResults()
    # results = store.collection.query(
    #     query_texts=[query], n_results=10
    # )
    # retrieved_documents = results['documents'][0]
    #
    # output = rag(query=query, retrieved_documents=retrieved_documents)
    # print(output)
    store = StoreResults()

    persistent_client = store._instance.chroma_client
    collection = store.collection

    vector_db = Chroma(
        client=persistent_client,
        collection_name="collection_name",
        embedding_function=EmbedChunks("all-MiniLM-L6-v2").embedding_model,
    )
    #
    # vector_db = Chroma(persist_directory="./chromadb",
    #                    embedding_function=EmbedChunks("all-MiniLM-L6-v2").embedding_model)

    # Run similarity search query
    q = "what are the strategies to mitigate hallucination in llm"
    v = vector_db.similarity_search(q, include_metadata=True)

    # Run the chain by passing the output of the similarity search
    chain = load_qa_chain(openai_client, chain_type="stuff")
    res = chain({"input_documents": v, "question": q})
    print(res["output_text"])
