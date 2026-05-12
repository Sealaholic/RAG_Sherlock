# Sherlock Holmes RAG System

A Retrieval-Augmented Generation (RAG) system built with Python for semantic search and question answering over Sherlock Holmes stories.

# Web Page

https://rag-sherlock.onrender.com/

## Features

* Document loading and preprocessing
* Text chunking for long documents
* TF-IDF based retrieval system
* Cosine similarity ranking
* OpenAI GPT-powered answer generation
* Local caching for processed chunks and vectors

## Tech Stack

* Python
* scikit-learn
* OpenAI API
* NumPy

## Project Structure

```bash
docs/           # Sherlock Holmes text files
rag_system.py   # Main RAG pipeline
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

Run the project:

```bash
python rag_system.py
```

## Example Questions

* Who is Professor Moriarty?
* Which stories involve disguises?
* What is Sherlock Holmes' relationship with Watson?

## Future Improvements

* OpenAI embedding-based retrieval
* pgvector integration
* Hybrid search (TF-IDF + embeddings)
* Web interface with FastAPI
* Citation-enhanced responses
