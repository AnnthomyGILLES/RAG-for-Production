import uuid
from pathlib import Path

import ray

from embeddings import EmbedChunks
from html_section_parser import extract_sections
from storage import StoreResults
from text_splitter import chunk_section

ray.init(ignore_reinit_error=True, num_gpus=None, num_cpus=4)


def add_unique_id(batch):
    batch["index"] = [f"{i}-{uuid.uuid4()}" for i in range(len(batch))]
    return batch


def get_html_files(directory, num_files):
    """
    Retrieve a list of HTML file paths up to a specified limit from a directory.
    """
    html_files = [path for path in directory.rglob("*.html") if not path.is_dir()]
    return html_files[:num_files]


def main():
    base_dir = Path(__file__).parent
    EFS_DIR = "desired/output/directory"
    DOCS_DIR = Path(base_dir, EFS_DIR, "docs.ray.io/en/master/")
    num_files_to_process = 4

    # Get a list of HTML file paths
    html_files = get_html_files(DOCS_DIR, num_files_to_process)

    # Create a Ray data set from the file paths
    ds = ray.data.from_items([{"path": path} for path in html_files])

    # Print the count of documents in the data set
    print(f"{ds.count()} documents")

    # Extract sections from a specific document and create a data set of sections
    sections_ds = ds.flat_map(extract_sections)

    return sections_ds


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
