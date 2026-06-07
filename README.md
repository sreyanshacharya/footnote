# Footnote

Footnote is a fully on-device, Retrieval-Augmented-Generation (RAG) study assistant that answers questions from user-uploaded PDFs using semantic search and a locally run LLM.
It features a decoupled microservices architecture with a high-performance FastAPI backend and a Streamlit frontend, fully containerized using Docker and Docker Compose for seamless deployment.

## Features

- Upload and index multiple PDFs concurrently
- Semantic search over document chunks using FAISS
- Locally run Phi-4-mini model (3.8B param) optimized with 4-bit (NF4) quantization
- Decoupled architecture separating the heavy tensor computations from the UI layout
- Sidebar Index Manager to visualize currently active files and completely wipe the vector index
- Context-aware question answering with PyTorch Scaled Dot Product Attention (SDPA) optimization
- Fully offline. No APIs, no cloud.

## Architecture & Pipeline

Footnote is structured as a decoupled multi-service system:
- **Frontend (Streamlit)**: Handles presentation layer logic, file upload buffers, sidebar state management, and user chat layouts.
- **Backend (FastAPI)**: Serves synchronous API worker threads for heavy text extraction, embedding matrix processing, and autoregressive LLM text token generation.

The execution pipeline follows a standard local RAG flow:
1. Uploaded PDFs are processed and text is extracted using pypdf.
2. The extracted text is broken down into fixed-size context-aware chunks.
3. Each chunk is embedded using a lightweight embedding model (all-MiniLM-L6-v2) mapped explicitly to the CPU to preserve dedicated VRAM.
4. These embeddings are converted to float32 arrays, L2 normalized, and indexed using FAISS.
5. When a user asks a question, the query is embedded and FAISS runs a vector similarity search to return the top-k relevant chunks.
6. Phi-4-mini is fed the explicit context alongside a custom attention mask tensor, generating a grounded response with minimal hallucination risk.

## Performance Profiling & Optimization 

This system is explicitly optimized and benchmarked to run stably on highly constrained consumer hardware (Tested on a 6GB VRAM RTX 4050 Mobile GPU). 

* **Concurrency & VRAM Stability:** The backend streaming endpoints are natively instrumented to track Time-To-First-Token (TTFT), Inter-Token Latency (ITL), and GPU memory deltas. Strict asynchronous thread joining and explicit tensor garbage collection ensure a `+0.00 MB` VRAM delta per request, completely neutralizing memory leaks under concurrent multi-user load.

* **Hardware-Aware Constraints:** Based on rigorous automated matrix testing, the retrieval limit is strictly capped at `k=3` chunks, and embeddings are isolated to the CPU. This prevents KV-Cache expansion from causing Out-Of-Memory (OOM) crashes and stops compute thrashing during autoregressive generation.

> **Read the Engineering Logs:** For a deep dive into the raw telemetry data, load testing results, and metric charts, view the `benchmarks/` directory:
> * [Day 1 - 3: Instrumentation & Concurrency VRAM Leaks](benchmarks/01_instrumentation_and_concurrency.md)
>
> * [Day 4 - 5: Matrix Benchmarking & KV-Cache Scaling](benchmarks/02_benchmark_matrix.md)

## Deployment with Docker

The easiest way to run the entire multi-service ecosystem is using Docker Compose. Ensure you have Docker and the NVIDIA Container Toolkit installed on your host machine for native GPU passthrough.

```bash
git clone https://github.com/sreyanshacharya/footnote.git
cd footnote
docker compose up --build
```

Once the containers are successfully running, open your browser to access the environments:
- **Frontend Application**: http://localhost:8501
- **Backend API Documentation**: http://localhost:8000/docs

## Local Installation (Alternative)
```bash
python -m venv footnote
footnote\Scripts\activate
pip install -r requirements.txt
```

## Running Locally

To run without containers, you must spin up both services concurrently in separate terminal sessions:

**Terminal 1 (Backend Core API):**
```bash
uvicorn api:app --reload
```

**Terminal 2 (Frontend Interface):**
```bash
streamlit run app.py
```

## Example use cases

- Exam revision from lecture notes
- Searching through technical PDFs
- Personal knowledge base QA
- Sensitive document querying
- Offline study assistant

## Notes

- Models run locally - no cloud APIs required
- Optimized for consumer GPUs via 4-bit quantization and double quantization to safely prevent Windows WDDM shared memory overflow (tested on RTX 4050)
- Persistent data volumes ensure FAISS indices, raw uploads, and HuggingFace cache directories are preserved across container cycles

## Tech Stack

**LLM & NLP**
- Phi-4-mini-instruct (local LLM)
- all-MiniLM-L6-v2 from sentence-transformers (Embedding Model)
- HuggingFace Transformers & Accelerate
- BitsAndBytes (4-bit NF4 quantization)
- PyTorch (SDPA implementation)

**Retrieval & API Backend**
- FAISS (vector similarity search)
- FastAPI (asynchronous backend framework layout)
- Uvicorn (ASGI server implementation)

**Document Processing**
- pypdf

**Application Frontend**
- streamlit

**Environment & DevOps**
- Docker & Docker Compose
- Python

## Author

### Sreyansh Acharya
- CSE(AI/ML) at GITAM Hyd
- Interests in Deep Learning, ML Systems, Astrophysics and Scientific Computing

## Connect with me :

- [linkedin](www.linkedin.com/in/sreyanshacharya)
- [github](https://github.com/sreyanshacharya)
- [gmail](sreyanshacharyaa@gmail.com)