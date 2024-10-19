# LinkGen-Insights

This project implements a system to extract, process, and analyze LinkedIn posts using advanced natural language processing and a Retrieval-Augmented Generation (RAG) architecture. It provides insights into LinkedIn posts by utilizing Qdrant for vector-based semantic search and Streamlit for an interactive user interface.

## Features

- Automated LinkedIn post extraction and processing
- Real-time data processing with Bytewax
- Scalable vector storage using Qdrant
- Query interface via Streamlit
- Advanced retrieval with Llama 3.1 LLM and LangChain

## Setup

### Prerequisites

- Docker
- Poetry

### Running Qdrant with Docker

To start the Qdrant vector storage, run the following command:

```bash
make run_qdrant_as_docker
```
This will set up and run Qdrant in a Docker container on port 6333.

Install Dependencies
To install the necessary dependencies using Poetry, run:


```bash
poetry install
```

Running the Application
Once Qdrant is up and dependencies are installed, you can run the application locally with:

```bash
poetry run streamlit run app.py
```
This will launch the interactive Streamlit interface for querying LinkedIn posts.

Technologies Used
Python
Qdrant for vector storage
LangChain and Llama 3.1 LLM for natural language processing
Streamlit for the user interface
Poetry for dependency management
