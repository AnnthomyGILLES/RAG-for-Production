import uuid
from pathlib import Path

import ray
from loguru import logger

from config import config
from documents_loader import ingest_documents
from embeddings import EmbedChunks
from splitter import chunk_section
from storage import StoreResults

ray.init(ignore_reinit_error=True, **config["ray_init"])


def add_unique_id(batch):
    batch["index"] = [f"{i}-{uuid.uuid4()}" for i in range(len(batch))]
    return batch


if __name__ == "__main__":
    # Directory path for the documents
    directory_path = Path(__file__).parent / config["directory_path"]

    # Ingest documents from the directory
    langchain_documents = ingest_documents(directory_path)

    # Create a dataset of documents
    sections_ds = ray.data.from_items(langchain_documents)

    # Log the number of documents
    logger.info(f"{sections_ds.count()} documents")

    # Create chunks from the documents
    chunks_ds = sections_ds.flat_map(chunk_section)

    # Embed the chunks
    fn_constructor_kwargs = {"model_name": config["models"]["embedding"]}
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs=fn_constructor_kwargs,
        concurrency=config["batch_processing"]["concurrency"],
    )

    # Add unique IDs to the dataset
    ray_dataset_with_index = embedded_chunks.map_batches(
        add_unique_id, batch_format="pandas"
    )

    # Index and store the data
    _ = ray_dataset_with_index.map_batches(
        StoreResults,
        batch_size=config["batch_processing"]["batch_size"],
        concurrency=config["batch_processing"]["concurrency"],
    ).count()
