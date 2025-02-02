# Information Retrieval Project

This project focuses on implementing and evaluating various information retrieval techniques using Python. The project includes scripts for indexing, query expansion, hybrid search, and semantic search.

## Project Structure

- **HYBRID-SEARCH.PY**: This script performs a hybrid search using both lexical and semantic search engines. It combines the results using the Reciprocal Rank Fusion method and evaluates the retrieval performance.

- **SEMANTIC-SEARCH.PY**: This script creates a collection in ChromaDB using the SentenceTransformer model and evaluates the retrieval performance.

- **QUERY-EXPANSION.PY**: This script evaluates the query expansion method using the Signed Log-Likelihood Ratio (LLR) method.

- **SIGNED-LLR.PY**: This script contains functions to compute term frequencies, calculate LLR, and expand queries using the Signed LLR method.

- **BM25.PY**: This script runs the BM25 algorithm for the TREC-COVID dataset with different configurations and evaluates the retrieval performance.

- **INDEXING.PY**: This script is used to index a corpus of documents using the BM25 algorithm.

## Setup

1. Clone the repository:
    ```
    git clone https://github.com/your-username/Information-Retrieval.git
    cd Information-Retrieval
    ```

## Disclaimer

This was a project developed for the Information Repositories course of the University of Oviedo.

Mark obtained: 9.8 points over 10 possible points.
