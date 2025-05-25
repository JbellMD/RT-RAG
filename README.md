# RT-RAG Assistant

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) <!-- Update if you chose a different license -->
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub Actions CI](https://github.com/JbellMD/RT-RAG/actions/workflows/python-app.yml/badge.svg)](https://github.com/JbellMD/RT-RAG/actions/workflows/python-app.yml) <!-- Replace JbellMD/RT-RAG with your actual repo path -->

**A Retrieval Augmented Generation (RAG) based assistant for querying your documents.**

This project provides a robust framework for building and deploying a RAG assistant. It allows users to ingest documents, create a searchable vector store, and query these documents using a natural language interface, leveraging large language models for response generation.

## Table of Contents

- [RT-RAG Assistant](#rt-rag-assistant)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Cloning the Repository](#cloning-the-repository)
    - [Setting up the Environment](#setting-up-the-environment)
    - [Installing Dependencies](#installing-dependencies)
    - [Environment Variables](#environment-variables)
  - [Usage](#usage)
    - [Running the API Server](#running-the-api-server)
    - [Example API Interaction](#example-api-interaction)
  - [Docker](#docker)
    - [Building the Docker Image](#building-the-docker-image)
    - [Running with Docker Compose](#running-with-docker-compose)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Methodology](#methodology)
  - [Performance](#performance)
  - [Contributing](#contributing)
  - [License](#license)
  - [Changelog](#changelog)
  - [Citation](#citation)
  - [Contact and Support](#contact-and-support)

## Features

*   Document ingestion from various formats (JSON, PDF, TXT - extendable).
*   Vector store creation and management using FAISS.
*   OpenAI integration for embeddings and language model capabilities.
*   FastAPI backend for serving the RAG assistant via a REST API.
*   Structured for production-readiness, including Dockerization, CI, and comprehensive documentation setup.

## Project Structure

A brief overview of the key directories:

```
RT-RAG/
├── .github/            # GitHub Actions workflows and issue templates
├── .vscode/            # VSCode settings (optional)
├── data/
│   ├── raw/            # Raw input data files
│   └── processed/      # Processed data (e.g., cleaned, transformed)
├── docs/               # Sphinx documentation source files
├── frontend-react/     # React frontend (if applicable)
├── requirements/       # Python dependency files (base, dev, test, docs)
├── scripts/            # Utility scripts (e.g., setup, data processing)
├── src/
│   └── rt_rag/         # Main Python package for the RAG assistant
│       ├── __init__.py
│       ├── api_main.py   # FastAPI application
│       └── rag_assistant.py # Core RAG logic
├── tests/
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
├── vectorstore/        # FAISS vector store (or other persistent stores)
├── .env                # Environment variables (local, not committed)
├── .env.example        # Example environment variables
├── .gitignore
├── .pre-commit-config.yaml # Pre-commit hook configurations
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── Dockerfile
├── docker-compose.yml
├── LICENSE
├── pyproject.toml      # Project metadata and build system configuration
├── README.md           # This file
└── setup.py            # Setup script for packaging
```

For more details, see the [Project Structure section in the documentation](./docs/project_structure.md)

## Installation

### Prerequisites

*   Python 3.9 or higher
*   Git
*   (Optional) Docker and Docker Compose

### Cloning the Repository

```bash
git clone https://github.com/JbellMD/RT-RAG.git 
cd RT-RAG
```

### Setting up the Environment

It is highly recommended to use a virtual environment:

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Installing Dependencies

Install the core dependencies:

```bash
pip install -r requirements/base.txt
```

To install the package itself (e.g., for command-line tools or if other local packages depend on it):

```bash
pip install .
```

For development, you might also want to install development and testing dependencies:

```bash
pip install -r requirements/dev.txt
pip install -r requirements/test.txt
```

### Environment Variables

Copy the example environment file and populate it with your actual credentials and settings:

```bash
cp .env.example .env
```

Edit `.env` with your details, especially `OPENAI_API_KEY`.

## Usage

### Running the API Server

Once dependencies are installed and your `.env` file is configured, you can run the FastAPI application using Uvicorn:

```bash
uvicorn src.rt_rag.api_main:app --reload
```

This will typically start the server at `http://127.0.0.1:8000`.

### Example API Interaction

(Provide examples of how to interact with your API endpoints, e.g., using `curl` or a Python script with `requests`.)

**Example: Querying the RAG assistant**

```bash
curl -X POST "http://127.0.0.1:8000/query/" \
     -H "Content-Type: application/json" \
     -d '{"query_text": "What are the main findings of project X?"}'
```

## Docker

### Building the Docker Image

To build the Docker image for this project:

```bash
docker build -t rt-rag-assistant .
```

### Running with Docker Compose

Docker Compose simplifies running the application and any related services.

```bash
docker-compose up
```

This will build the image (if not already built) and start the `rag-assistant` service, making it available on port 8000.
Your `.env` file will be used for environment variables within the container.

## Testing

To run the test suite (unit and integration tests):

1.  Ensure you have test dependencies installed:
    ```bash
    pip install -r requirements/test.txt
    ```
2.  Run pytest from the root directory:
    ```bash
    pytest tests/
    ```

To include coverage reports:

```bash
pytest tests/ --cov=src/rt_rag --cov-report=html # or xml, term
```

## Documentation

Comprehensive documentation is available and can be built using Sphinx.

1.  Install documentation dependencies:
    ```bash
    pip install -r docs/requirements.txt
    ```
2.  Build the HTML documentation:
    ```bash
    cd docs
    make html
    ```
3.  Open `docs/_build/html/index.html` in your browser.

(Alternatively, link to your ReadTheDocs or GitHub Pages site if you deploy it there.)

## Methodology

This project implements a Retrieval Augmented Generation (RAG) pipeline. Key steps include:

1.  **Document Loading**: Ingesting source documents.
2.  **Text Splitting**: Breaking down documents into manageable chunks.
3.  **Embedding Generation**: Converting text chunks into vector embeddings using models like OpenAI's.
4.  **Vector Storage**: Storing embeddings in a FAISS vector store for efficient similarity search.
5.  **Retrieval**: Given a user query, retrieve relevant document chunks from the vector store.
6.  **Response Generation**: Feed the retrieved context and the original query to a large language model (LLM) to generate a coherent answer.

More detailed information can be found in the [project documentation](./docs/introduction.md).

## Performance

(This section should be updated with performance metrics once available. Consider aspects like query latency, document ingestion speed, and resource utilization.)

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](./CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details. <!-- Update if you chose a different license -->

## Changelog

See the [CHANGELOG.md](./CHANGELOG.md) for a history of changes to this project.

## Citation

If you use this project in your research or work, please consider citing it:

```bibtex
@software{yourusername_rt_rag_assistant_2024,
  author = {Your Name/Organization},
  title = {{RT-RAG Assistant}},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/JbellMD/RT-RAG}} 
}
```

## Contact and Support

*   If you have questions, encounter issues, or want to contribute, please open an issue on the [GitHub Issues page](https://github.com/JbellMD/RT-RAG/issues).
*   For other inquiries, contact [Your Name/Email] (optional).
