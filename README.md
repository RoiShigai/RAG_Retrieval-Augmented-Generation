# Rage Against The Machin RAG

## Table of Content

- Description
- Instruction
- System Architecture
- Chunking Strategy
- Retrieval Method
- Ressources

## Description

Rage Against The Machine is a RAG discovery implementation project from 42 School. We had to implement a Retrieved Augmented Generation pipeline on the Open Ai vllm corpus for Markdown an Python files. The return Answered is produced by Qwen0.6b.

### Features

- Incremental Index and files versionning tracking
- SQLite database for better Indexing, faster retrieval, versionning and memory performance
- Caching system

## Instruction

### Dependencies

- fire 
- flake8
- mypy
- pydantic
- tqdm

### Installation

The repository contain a 'uv.lock' file to simplify to simplify the installation to one command:
```bash
uv sync
```

## System Architecture

### SQLite Database

An SQLite database is implemented in this project. This data base is required to optimize data storing and data acquisition due to high number of chunks created during the corpus Indexing and also for better file versionning.

The database contain multiple tables:
- **Files table**: containing the path to the file, hased content signature, timestamp metadata. This table is used for files versionning.
- **Chunks table**: Containing all of the chunks, their metadata(id, lengths, number of tokens...) used to reconstruct the chunks and for BM25 calculation.
- **Chunk_tokens table**: Table containing every notable tokens, their associated chunks and some stats for faster BM25 calculation (token frequency into a chunk)
- **token_stats table**: Corresponding of the token frequency into the whole chunks
- **index_metadata table**: Table containing the number of chunks and the total tokens lengths for faster BM25 calculation.
- **database_metadata table**: Contain database metadata to verify the version of the database for the cached data.

All of this table are used for improve the performance of the BM25 calculation and chunks retrieving. BM25 can quickly be calculated using the chunks, tokens, tokens_stats metadata without reconstructing the whole BM25. 
High number of random query to the database can slow and erase the performance gains of this feature, so most of query are batched together into big query to preserve the performance of database access.

### File versioning

The file versioning feature has been added by storing the files metadata into the database. Each file during the first Indexing have their content hashed using SHA-256, and their metadata (last modification timestamp) store into the database. 
If a file has their timestamp modified, the RAG will hash the content of the file and compare the resulting signature with the stored value. If hashed signature are identical then the stored modified timestamp is updated otherwise it means that the content has change so the stored chunks corresponding to this file can be outdate. 
To fix this, the RAG will retrieve the corresponding file chunks and replace them by the new generated chunks. 

This feature is not launched automatically at each search due to project requirement. To register the new data, a new Indexing command must be launch by the user. In real RAG production, a system that periodically check the last Indexing date can automatize this auto updating system, depending of the document nature and company activity, this system could be set every day/weeks or month.

This feature increase by a lot the first Indexing time (about ~2-3min) but after that the Indexing updates can takes just a few seconds (10-20 sec) depending of the  number of file to update.

### Caching System

RAG Project contain a caching system for faster common question retrieving and to diminish the query to the database. This caching systems works by storing the MinimalSearchResult into a JSON file, the json filename is hashed user query using SHA-256 hashing algorithm. 

Each cached response obey to a pydantic BaseModel containing some metadata for response validity (database version, the top-k results chunks and the user query). If the database version or top-k results number doesn't correspond to the cached data, the RAG will use the pipeline standard pipeline and replace the cached data with new valid data.

This cached feature help us to gain significant performance, about ~20% better response time.

## Chunking Strategy

This project use 2 different chunking strategy for MarkDown files and Python files.

### Mardown files

Mardown Files if well structured are easy to chunks, we can separate each part using the header contained in the file then separate those header by the paragraph contained in the Hearder Section. The paragraph are then chunked using an overlap to keep semantic between two chunks. The Markdown Chunker try to be sentence and token aware, avoiding to split mid-sentence or token in half. If the max_chunk_size is extremelly low splitting sentence/token is impossible to avoid then it will respect the chunk size by splitting the token or the sentence in half. This decision is made for purely experimentation with chunk size, the output for those case are expected to be false as the semantic within the chunks is shredded by extremely low size.

### Python files

Python files are chunked using another technique for demonstration. The algorithm choosed is based on the builtin python AST to improve semantic and python code chunking. each Class are chunked into Class part, respecting method definition and each functions defintion are dispatch into their respective chunks. If the max chunk size value is to low then it authorized token split and function/class definition splitting. Into each Chunks token list, the tokens "class", "function" are injected to improve chance to get the right type of chunks from the user request. ex: "which class contain the method to setup vllm ?". 

# Resources
- [cAST Paper](https://arxiv.org/abs/2506.15655?utm_source=chatgpt.com): Research Paper by 'Yilin Zhan' for Code retrieval Strategy
- [Empirical Study](https://arxiv.org/abs/2605.04763?utm_source=chatgpt.com): Controlled Empirical Study by 'Xinjian Wu' teams from King's College London, about Chunking Strategy for Retrieval-Augmented Code Completion
- [RAG techniques Evaluation](https://arxiv.org/pdf/2407.01219): Research paper that compare different RAG techniques for Indexing/Ranking and Retrieving
- [LangChain LLM Application Deployment tutorial](https://apxml.com/courses/langchain-production-llm): LangChain tutorial for LLM deployment and has a special chapter for RAG deployment
