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

## Deployment with Docker

The easiest way to run the entire multi-service ecosystem is using Docker Compose. Ensure you have Docker and the NVIDIA Container Toolkit installed on your host machine for native GPU passthrough.

```bash
git clone [https://github.com/sreyanshacharya/footnote.git](https://github.com/sreyanshacharya/footnote.git)
cd footnote

# Build and launch both services simultaneously
docker compose up --build