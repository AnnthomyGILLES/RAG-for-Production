import chromadb


class StoreResults:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chromadb")
        self.collection = self.chroma_client.get_or_create_collection(
            name="ray-documentation"
        )

    def __call__(self, batch):
        # Prepare the data for insertion
        documents = []
        for text, source, embedding in zip(
            batch["text"], batch["source"], batch["embeddings"]
        ):
            # Adapt this part based on how ChromaDB expects the data
            document = {"text": text, "source": source, "embedding": embedding}
            documents.append(document)
        self.collection.upsert(documents)
        return {}


if __name__ == "__main__":
    sample_sections = [
        {
            "text": "Section 1 text goes here. More details about section 1.",
            "source": "Source 1",
        },
        {
            "text": "Section 2 has different content. It might be longer or shorter.",
            "source": "Source 2",
        },
        {
            "text": "Another section, Section 3, with its own unique text and information.",
            "source": "Source 3",
        },
    ]

    StoreResults()
