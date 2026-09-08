# VectorStore Engine

An end-to-end, zero-dependency Vector Database Engine built in Python. Features include exact k-NN search, Hierarchical Navigable Small World (HNSW) Approximate Nearest Neighbor (ANN) search, BM25 probabilistic keyword retrieval, Reciprocal Rank Fusion (RRF) hybrid search, FastAPI REST API, interactive CLI, Docker support, and full CI/CD.

---

## System Architecture

+-------------------------------------------------------------------------+
|                               Client Layer                              |
|   +-----------------------+     +------------------+     +----------+   |
|   |  Interactive CLI      |     |  HTTP REST API   |     |  Locust  |   |
|   |  (cli.py)             |     |  (app.py)        |     |  Load    |   |
|   +-----------+-----------+     +--------+---------+     +----+-----+   |
+---------------+--------------------------|--------------------+---------+
                |                          |                    |
                +-----------------+--------+--------------------+
                                  |
+---------------------------------v---------------------------------------+
|                            VectorStore Core                             |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |                        Indexing & Storage                         |  |
|  |  +----------------------+  +-------------------+  +------------+  |  |
|  |  | In-Memory Vector     |  | Inverted Index    |  | Metadata   |  |  |
|  |  | Storage Dict         |  | (Term Freq & IDF) |  | Store      |  |  |
|  |  +----------+-----------+  +---------+---------+  +-----+------+  |  |
|  +-------------|------------------------|------------------|---------+  |
|                |                        |                  |            |
|  +-------------v------------------------v------------------v----------+  |
|  |                       Retrieval Backends                           |  |
|  |  +----------------------+  +-------------------+  +------------+  |  |
|  |  | Exact Search (k-NN)  |  | HNSW Graph Engine |  | BM25       |  |  |
|  |  | Cosine/Euclidean/IP  |  | (Multi-layer ANN) |  | Keyword    |  |  |
|  |  +----------+-----------+  +---------+---------+  +-----+------+  |  |
|  +-------------|------------------------|------------------|---------+  |
|                |                        |                  |            |
|  +-------------v------------------------v------------------v----------+  |
|  |                       Fusion & Operations                          |  |
|  |  +--------------------------------------------------------------+  |  |
|  |  | Reciprocal Rank Fusion (RRF) Hybrid Search Engine            |  |  |
|  |  +--------------------------------------------------------------+  |  |
|  |  +--------------------------------------------------------------+  |  |
|  |  | JSON Persistence (save_to_json / load_from_json)               |  |  |
|  |  +--------------------------------------------------------------+  |  |
|  +------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+

---

## Core Features

* **Multiple Search Modes:**
  * **Dense Vector Search:** Exact brute-force k-NN using Cosine Similarity, Euclidean Distance, or Inner Product.
  * **HNSW ANN Search:** Multi-layer graph index for sub-linear similarity search across high-dimensional vector spaces.
  * **BM25 Keyword Search:** Full-text probabilistic ranking over metadata fields with inverted index indexing.
  * **RRF Hybrid Search:** Rank-based fusion combining dense semantic search and sparse BM25 keyword matching.
* **REST API:** High-performance REST endpoints built with FastAPI, Pydantic validation, and interactive Swagger documentation.
* **Interactive CLI:** Terminal application supporting batch insertions, state persistent saving/loading, and inline benchmarking.
* **Containerized:** Production-ready Dockerfile and docker-compose.yml with automated health checks.
* **CI/CD & Testing:** Automated linting (ruff), unittest coverage for CLI and API, and load testing via Locust and httpx.

---

## Repository Structure

.
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow
├── app.py                     # FastAPI REST API implementation
├── benchmark_api.py           # Async performance benchmarking script
├── cli.py                     # Interactive CLI application
├── docker-compose.yml         # Multi-container orchestration
├── Dockerfile                 # Container image specification
├── hnsw.py                    # Hierarchical Navigable Small World index
├── locustfile.py              # Locust load testing scenario
├── README.md                  # System documentation
├── requirements.txt           # Python dependencies
├── test_app.py                # FastAPI integration test suite
├── test_cli.py                # CLI unit test suite
├── test_vector_store.py       # Core engine unit test suite
└── vector_store.py            # Primary VectorStore implementation

---

## Quick Start

### 1. Local Python Setup

git clone https://github.com/your-username/vector-store.git
cd vector-store

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m unittest discover -s . -p "test_*.py"

### 2. Run the REST API

uvicorn app:app --reload --port 8000

Access Swagger UI documentation at http://localhost:8000/docs.

### 3. Run with Docker Compose

docker compose up --build -d
curl http://localhost:8000/

### 4. Run Interactive CLI

python cli.py

---

## API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| / | GET | API status, document count, and backend configuration |
| /vectors | POST | Insert single vector document |
| /vectors/batch | POST | Bulk insert multiple vector documents |
| /vectors/{id} | GET | Retrieve record by ID |
| /vectors/{id} | PUT | Update vector or metadata by ID |
| /vectors/{id} | DELETE | Delete vector document by ID |
| /search/dense | POST | Dense similarity search (exact or hnsw backends) |
| /search/keyword | POST | BM25 probabilistic keyword search |
| /search/hybrid | POST | Reciprocal Rank Fusion (RRF) hybrid search |
| /store/save | POST | Save store state to JSON file |
| /store/load | POST | Load store state from JSON file |

---

