import uuid
from pathlib import Path

import ray
from loguru import logger

from documents_loader import ingest_documents
from embeddings import EmbedChunks
from splitter import chunk_section
from storage import StoreResults

ray.init(ignore_reinit_error=True, num_gpus=None, num_cpus=4)


def add_unique_id(batch):
    batch["index"] = [f"{i}-{uuid.uuid4()}" for i in range(len(batch))]
    return batch


def main():
    directory_path = Path(__file__).parent / "data" / "raw"

    langchain_documents = ingest_documents(directory_path)

    ds = ray.data.from_items(langchain_documents)

    logger.info(f"{ds.count()} documents")
    return ds


if __name__ == "__main__":
    sections_ds = main()

    # Define chunking parameters
    chunk_size = 300
    chunk_overlap = 50
    separators = ["\n\n", "\n", " ", ""]

    # Create chunks dataset
    chunks_ds = sections_ds.flat_map(chunk_section)

    # Embed chunks
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": "text-embedding-ada-002"},
        concurrency=1,
    )

    ray_dataset_with_index = embedded_chunks.map_batches(
        add_unique_id, batch_format="pandas"
    )

    # Index data
    _ = ray_dataset_with_index.map_batches(
        StoreResults,
        batch_size=128,
        concurrency=1,
    ).count()
