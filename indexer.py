from functools import partial
from pathlib import Path

import ray

from embeddings import EmbedChunks
from html_section_parser import extract_sections
from text_splitter import chunk_section

ray.init(ignore_reinit_error=True, num_gpus=None, num_cpus=1)


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
    chunks_ds = sections_ds.flat_map(
        partial(chunk_section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )
    # l = chunks_ds.take_all()

    # Embed chunks
    embedded_chunks = chunks_ds.map_batches(
        EmbedChunks,
        fn_constructor_kwargs={"model_name": "text-embedding-ada-002"},
        concurrency=1,
    )
    # l = embedded_chunks.take_all()
    # print(l)
    # Index data
    # embedded_chunks.map_batches(
    #     StoreResults,
    #     concurrency=1
    # ).count()
