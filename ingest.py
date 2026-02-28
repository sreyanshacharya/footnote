import pypdf
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer


def extract(pdf):
  reader = pypdf.PdfReader(pdf)
  extractedtext = ""
  for page in reader.pages:
    text=page.extract_text()
    if text:
      extractedtext+=text
  return extractedtext

def chunk_text(text, size=232):
  chunks = []
  words = text.split()
  for i in range(0, len(words), size):
    curr_chunk = words[i:i+size]
    chunks.append(" ".join(curr_chunk))
  return chunks

def ingest(data_dir="data", db_dir="model-files"):
  data_path = Path(data_dir)
  pdfcount = 0
  big_text = ""

  #extract text
  for item in data_path.iterdir():
    if item.is_file():
      if item.suffix == '.pdf':
        big_text+=extract(item.as_posix())
        pdfcount+=1

  #chunkify
  chunks = chunk_text(big_text)

  #embedding
  model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
  embeddings = model.encode(chunks)


  # float32 and normalisation
  embeddings=embeddings.astype("float32")
  faiss.normalize_L2(embeddings)

  #faiss index
  index = faiss.IndexFlatIP(384)
  index.add(embeddings)

  Path(db_dir).mkdir(exist_ok=True)

  # saving the memory
  faiss.write_index(index, "model-files/index.faiss")
  with open("model-files/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

  return pdfcount, len(chunks)