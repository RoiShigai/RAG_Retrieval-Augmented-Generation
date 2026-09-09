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

## Chunking Strategy

This project use 2 different chunking strategy for MarkDown files and Python files.

### Mardown files

Mardown Files if well structured are easy to chunks, we can separate each part using the header contained in the file then separate those header by the paragraph contained in the Hearder Section. The paragraph are then chunked using an overlap to keep semantic between two chunks. The Markdown Chunker try to be sentence and token aware, avoiding to split mid-sentence or token in half. If the max_chunk_size is extremelly low splitting sentence/token is impossible to avoid then it will respect the chunk size by splitting the token or the sentence in half. This decision is made for purely experimentation with chunk size, the output for those case are expected to be false as the semantic within the chunks is shredded by extremely low size.

### Python files

Python files are chunked using another technique for demonstration. The algorithm choosed is based on the builtin python ast to improve semantic and python code chunking. each Class are chunked into Class part, respecting method definition and each functions defintion are dispatch into their respective chunks. If the max chunk size value is to low then it authorized token split and function/class definition splitting. Into each Chunks token list, the tokens "class", "function" are injected to improve chance to get the right type of chunks from the user request. ex: "which class contain the method to setup vllm ?". 

# Resources
- [cAST Paper](https://arxiv.org/abs/2506.15655?utm_source=chatgpt.com): Research Paper by 'Yilin Zhan' for Code retrieval Strategy
- [Empirical Study](https://arxiv.org/abs/2605.04763?utm_source=chatgpt.com): Controlled Empirical Study by 'Xinjian Wu' teams from King's College London, about Chunking Strategy for Retrieval-Augmented Code Completion
- [RAG techniques Evaluation](https://arxiv.org/pdf/2407.01219): Research paper that compare different RAG techniques for Indexing/Ranking and Retrieving
