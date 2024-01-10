import os

from dotenv import load_dotenv
from openai import OpenAI

from storage import StoreResults

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API"))


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
    query = "what are the strategies to mitigate hallucination in llm"
    store = StoreResults()
    results = store.collection.query(
        query_texts=[query], n_results=10
    )
    retrieved_documents = results['documents'][0]

    output = rag(query=query, retrieved_documents=retrieved_documents)
    print(output)
