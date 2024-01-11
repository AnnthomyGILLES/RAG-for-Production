import os

import numpy as np
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from storage import StoreResults

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API"))
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

MODEL = "gpt-3.5-turbo-1106"


def word_wrap(string, n_chars=72):
    # Wrap a string at the next space after n_chars
    if len(string) < n_chars:
        return string
    else:
        return (
            string[:n_chars].rsplit(" ", 1)[0]
            + "\n"
            + word_wrap(string[len(string[:n_chars].rsplit(" ", 1)[0]) + 1 :], n_chars)
        )


def project_embeddings(embeddings, umap_transform):
    umap_embeddings = np.empty((len(embeddings), 2))
    for i, embedding in enumerate(tqdm(embeddings)):
        umap_embeddings[i] = umap_transform.transform([embedding])

    return umap_embeddings


def rag(query, retrieved_documents, model=MODEL):
    information = "\n\n".join(retrieved_documents)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert research assistant  in artificial intelligence. Your users are asking questions about information contained in a documentation."
            "You will be shown the user's question, and the relevant information from documentation. Answer the user's question using only this information.",
        },
        {
            "role": "user",
            "content": f"Question: {query}. \n Information: {information}",
        },
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    content = response.choices[0].message.content
    return content


def augment_query_generated(query, model=MODEL):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert research in artificial intelligence. Provide an example answer to the given question, that might be found in a document like a scientific article. ",
        },
        {"role": "user", "content": query},
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content


def augment_multiple_query(query, model=MODEL):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert research assistant in artificial intelligence. Your users are asking questions about scientific article. "
            "Suggest up to five additional related questions to help them find the information they need, for the provided question. "
            "Suggest only short questions without compound sentences. Suggest a variety of questions that cover different aspects of the topic."
            "Make sure they are complete questions, and that they are related to the original question."
            "Output one question per line. Do not number the questions.",
        },
        {"role": "user", "content": query},
    ]

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    content = content.split("\n")
    return content


if __name__ == "__main__":
    import umap
    import matplotlib.pyplot as plt

    original_query = "what are the strategies to mitigate hallucination in llm"
    hypothetical_answer = augment_query_generated(original_query)

    joint_query = f"{original_query} {hypothetical_answer}"
    print(word_wrap(joint_query))

    store = StoreResults()
    results = store.collection.query(
        query_texts=[joint_query], n_results=5, include=["documents", "embeddings"]
    )
    retrieved_documents = results["documents"][0]

    output = rag(query=original_query, retrieved_documents=retrieved_documents)
    print("-" * 30)
    print(word_wrap(output))

    embeddings = store.collection.get(include=["embeddings"])["embeddings"]
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

    retrieved_embeddings = results["embeddings"][0]
    original_query_embedding = embedding_function([original_query])
    augmented_query_embedding = embedding_function([joint_query])

    projected_original_query_embedding = project_embeddings(
        original_query_embedding, umap_transform
    )
    projected_augmented_query_embedding = project_embeddings(
        augmented_query_embedding, umap_transform
    )
    projected_retrieved_embeddings = project_embeddings(
        retrieved_embeddings, umap_transform
    )

    plt.figure()
    plt.scatter(
        projected_dataset_embeddings[:, 0],
        projected_dataset_embeddings[:, 1],
        s=10,
        color="gray",
    )
    plt.scatter(
        projected_retrieved_embeddings[:, 0],
        projected_retrieved_embeddings[:, 1],
        s=100,
        facecolors="none",
        edgecolors="g",
    )
    plt.scatter(
        projected_original_query_embedding[:, 0],
        projected_original_query_embedding[:, 1],
        s=150,
        marker="X",
        color="r",
    )
    plt.scatter(
        projected_augmented_query_embedding[:, 0],
        projected_augmented_query_embedding[:, 1],
        s=150,
        marker="X",
        color="orange",
    )

    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"{original_query}")
    plt.axis("off")
    plt.savefig("my_plot.png")
