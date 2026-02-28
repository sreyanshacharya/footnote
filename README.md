# Footnote

Footnote is a fully on-device, Retrieval-Augmented-Generation (RAG) based study assistant that answers questions from user-uploaded PDFs using semantic search and a locally run LLM.
It runs entirely offline on consumer hardware and supports dynamic document ingestion through a simple StreamLit interface.

## Features

- Upload and index PDFs
- Semantic search over document chunks using FAISS
- Locally run Phi-4-mini model (3.8B param) for grounded answers
- Simple and functional StreamLit UI
- Context-aware question answering
- Fully offline. No APIs, no cloud.

## Pipeline

Footnote uses a standard local RAG pipeline :

1. Uploaded PDFs are processed and text is extracted.
2. The extracted text is broken down into fixed size context-aware chunks.
3. Embedding of these chunks is done through an embedding model (all-MiniLM-L6-v2)
4. These embeddings are stored in an index for quick retrieval using FAISS.
5. When a user asks a question, the query is embedded and FAISS returns the most similar chunks, which are used as context.
6. Phi-4 is prompted with the query and context and the generated answer is displayed back to the user.

The LLM is constrained to retrieved context only, ensuring grounded answers and very low risk of hallucination.

## Installation

```
git clone https://github.com/sreyanshacharya/footnote.git
cd footnote

python -m venv footnote
footnote\Scripts\activate

pip install -r requirements.txt
```

## Running

```
streamlit run app.py
```

## Example use cases

- Exam revision from lecture notes
- Searching through technical PDFs
- Personal knowledge base QA
- Sensitive document querying
- Offline study assistant

## Notes

- Models run locally - no external APIs required
- Optimized for consumer GPUs (tested on RTX 4050)
- Index is rebuilt for every new document ingested

## Tech Stack

**LLM & NLP**

- Phi-4-mini-instruct (local LLM)
- MiniLM-L6-v2 from sentence-transformers (Embedding Model)
- HuggingFace Transformers
- PyTorch

**Retrieval**

- FAISS (vector similarity search)

**Document Processing**

- pypdf

**Application**

- Streamlit

**Environment**

- Python, venv

## Author

- Sreyansh Acharya
  2nd Year CSE(ai/ml) at GITAM Hyd
  Interests in Deep Learning, Computer Vision, Astronomy, Scientific Computing

- ## Connect with me :
  - [linkedin](www.linkedin.com/in/sreyanshacharya)
  - [github](https://github.com/sreyanshacharya)
  - [gmail](sreyanshacharyaa@gmail.com)
