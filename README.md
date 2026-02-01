# RAG-Pipeline

**Retrieval-Augmented Generation (RAG) Pipeline** is a lightweight Python project that empowers you to build a simple RAG system — ingest documents, convert them into vector embeddings, store them in a vector database, and query them with natural language.

This project lays the foundation for knowledge-grounded LLM applications where an LLM can answer questions via context retrieved from your own custom datasets.

---

## Features

- Ingest and embed documents into a vector store  
- Perform semantic search over custom data  
- Query data with natural language  
-  Extensible for any RAG use-case

---

## Project Contents

| File / Folder | Purpose |
|---------------|---------|
| `data/` | Store documents to ingest (PDFs, text files, etc.) |
| `create_vector_database.py` | Build vector embeddings & store them |
| `query_data.py` | Query the vector database |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore settings |

---

## What is RAG?

**Retrieval-Augmented Generation (RAG)** combines traditional retrieval systems with Large Language Models (LLMs). Instead of relying on just a pretrained model’s memory, RAG systems **retrieve context from external data sources**, and feed that into the LLM to generate more grounded, accurate answers. :contentReference[oaicite:1]{index=1}

This architecture:
- improves factual accuracy
- reduces hallucinations
- enables reasoning over custom/private knowledge

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python **3.8+**
- `pip` (Python package manager)
- A terminal / command line

---

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Puneeth-24/RAG-Pipeline.git
   cd RAG-Pipeline
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate     # macOS / Linux
   venv\Scripts\activate        # Windows
   ```
3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```


### Running the Pipeline

1. Create Vector Database

Populate your data/ folder with documents you want to search over, then run:
```bash
python create_vector_database.py
```
This script will:
- read files from data/
- embed them into vector embeddings
- store those in the vector database

2. Query the Database

Ask questions using natural language:
```bash
python query_data.py
```
Enter your prompt when asked — the system will retrieve relevant context and return an answer grounded in your data.


## How It Works(High-Level)

1. Document Ingestion — Load documents from data/
2. Embeddings — Convert text to vector embeddings
3. Store in a Vector DB — Build a searchable vector index
4. Answer Queries — Retrieve relevant chunks and generate an answer

##  Example Use Cases

-  Build a Q&A bot over PDF reports
- Search and summarize company knowledge bases
-  Answer user queries using domain-specific data
-  Integrate with LLMs for customer support assistants

## Contributing

Contributions are welcome! Feel free to:
- improve scripts
- add file parsers (PDF, DOCX, media)
- integrate with advanced LLMs (OpenAI, HuggingFace, etc.)
- enhance the query interface

## Acknowledgements

Built with inspiration from various RAG pipelines shared in the community.
