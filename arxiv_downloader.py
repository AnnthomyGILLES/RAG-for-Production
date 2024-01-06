import time
from pathlib import Path

import arxiv
from tqdm import tqdm

if __name__ == '__main__':

    output_directory = Path.cwd() / "data" / "raw"
    # Construct the default API client.
    client = arxiv.Client()

    # Search for the 10 most recent articles matching the keyword "quantum."
    search = arxiv.Search(
        query="large language models",
        max_results=2,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )

    search_results = list(client.results(search))

    # for r in client.results(search):
    #     print(r.title)

    for result in tqdm(search_results):
        result.download_pdf(dirpath=output_directory)
        time.sleep(5)
