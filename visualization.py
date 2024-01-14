import umap
from matplotlib import pyplot as plt
from chromadb.utils import embedding_functions
from config import config
from generation import ResearchAssistant
from tools import project_embeddings


def compute_embeddings(texts, model_name):
    """
    Compute embeddings for a list of texts using a specified embedding model.

    This function initializes an embedding model based on the specified `model_name`
    and applies it to a list of texts to generate their embeddings.

    Args:
        texts (list of str): A list of texts for which embeddings are to be computed.
        model_name (str): The name of the model to be used for generating embeddings.

    Returns:
        np.ndarray: A numpy array containing the computed embeddings for the input texts.
    """
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    return embedding_function(texts)


# Function to perform UMAP transformation
def umap_transform_embeddings(embeddings):
    """
    Apply UMAP transformation to a set of embeddings.

    This function creates and fits a UMAP transformer with a fixed random state and
    transform seed, and then applies it to project the given embeddings into a lower-dimensional space.

    Args:
        embeddings (np.ndarray): A numpy array containing the embeddings to be transformed.

    Returns:
        np.ndarray: A numpy array containing the UMAP-transformed embeddings.
    """
    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
    return project_embeddings(embeddings, umap_transform)


# Function for plotting
def plot_embeddings(
    dataset_embeddings,
    retrieved_embeddings,
    original_query_embedding,
    augmented_query_embedding,
    original_query,
):
    """
    Plot and save a visualization of different sets of embeddings.

    This function plots the dataset embeddings, retrieved embeddings, original query embedding,
    and augmented query embedding in different colors and markers. It saves the plot as 'my_plot.png'.

    Args:
        dataset_embeddings (np.ndarray): Embeddings of the dataset to be plotted.
        retrieved_embeddings (np.ndarray): Embeddings of the retrieved documents.
        original_query_embedding (np.ndarray): Embedding of the original query.
        augmented_query_embedding (np.ndarray): Embedding of the augmented query.
        original_query (str): The original query text to be used as the plot title.

    """
    plt.figure()
    plt.scatter(dataset_embeddings[:, 0], dataset_embeddings[:, 1], s=10, color="gray")
    plt.scatter(
        retrieved_embeddings[:, 0],
        retrieved_embeddings[:, 1],
        s=100,
        facecolors="none",
        edgecolors="g",
    )
    plt.scatter(
        original_query_embedding[:, 0],
        original_query_embedding[:, 1],
        s=150,
        marker="X",
        color="r",
    )
    plt.scatter(
        augmented_query_embedding[:, 0],
        augmented_query_embedding[:, 1],
        s=150,
        marker="X",
        color="orange",
    )
    plt.gca().set_aspect("equal", "datalim")
    plt.title(f"{original_query}")
    plt.axis("off")
    plt.savefig("my_plot.png")


def main():
    """
    Main function to demonstrate the process of computing, transforming, and plotting embeddings.

    This function represents a workflow where an original query is augmented, embeddings are computed
    and transformed using UMAP, and finally, these embeddings are plotted for visual analysis.
    """
    original_query = "what are the strategies to mitigate hallucination in llm"
    model_name = config["models"]["embedding"]

    assistant = ResearchAssistant()
    joint_query = assistant.augment_query(original_query)

    results = assistant.store.collection.query(
        query_texts=[joint_query], n_results=5, include=["documents", "embeddings"]
    )

    embeddings = assistant.store.collection.get(include=["embeddings"])["embeddings"]
    projected_dataset_embeddings = umap_transform_embeddings(embeddings)

    retrieved_embeddings = results["embeddings"][0]
    original_query_embedding = compute_embeddings([original_query], model_name)
    augmented_query_embedding = compute_embeddings([joint_query], model_name)

    projected_original_query_embedding = project_embeddings(
        original_query_embedding, umap_transform_embeddings
    )
    projected_augmented_query_embedding = project_embeddings(
        augmented_query_embedding, umap_transform_embeddings
    )
    projected_retrieved_embeddings = project_embeddings(
        retrieved_embeddings, umap_transform_embeddings
    )

    plot_embeddings(
        projected_dataset_embeddings,
        projected_retrieved_embeddings,
        projected_original_query_embedding,
        projected_augmented_query_embedding,
        original_query,
    )


if __name__ == "__main__":
    main()
