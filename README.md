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

# Resources
- [cAST Paper](https://arxiv.org/abs/2506.15655?utm_source=chatgpt.com): Research Paper by 'Yilin Zhan' for Code retrieval Strategy
- [Empirical Study](https://arxiv.org/abs/2605.04763?utm_source=chatgpt.com): Controlled Empirical Study by 'Xinjian Wu' teams from King's College London, about Chunking Strategy for Retrieval-Augmented Code Completion
- [RAG techniques Evaluation](https://arxiv.org/pdf/2407.01219): Research paper that compare different RAG techniques for Indexing/Ranking and Retrieving
