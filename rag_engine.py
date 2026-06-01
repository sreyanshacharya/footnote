import faiss
import os
import pickle
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


#load embedding model
emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

#load index and chunks
index_path = "model_files/index.faiss"
chunks_path = "model_files/chunks.pkl"
index_exists = False
chunks_exists = False
if Path(index_path).is_file():
  index_exists = True

if Path(chunks_path).is_file():
    chunks_exists = True

#load llm and tokenizer
quantization_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype=torch.bfloat16
)

model_name = "microsoft/Phi-4-mini-instruct"
llm = AutoModelForCausalLM.from_pretrained(
  model_name,
  quantization_config=quantization_config,
  device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#query retrieval
def retrieve(question):
  if(Path(index_path).is_file() and Path(chunks_path).is_file()):
    index = faiss.read_index("model_files/index.faiss")
    with open("model_files/chunks.pkl", "rb") as f:
      chunks = pickle.load(f)

    qvec = emb_model.encode([question]).astype("float32")
    faiss.normalize_L2(qvec)
    #index search
    D, I = index.search(qvec, k=3)
    context = "\n\n".join([chunks[idx] for idx in I[0]])
    return context
  else:
    import os
    print(f"\n--- debug engine room ---")
    print(f"current working directory: {os.getcwd()}")
    print(f"looking for index at: {Path(index_path).absolute()} | exists: {Path(index_path).exists()}")
    print(f"looking for chunks at: {Path(chunks_path).absolute()} | exists: {Path(chunks_path).exists()}")
    print(f"-------------------------\n")
    raise RuntimeError("missing faiss index or chunks files on disk.")
  
#answer generation
def generate(context, question):
  messages = [
    {"role": "system", "content" : "You are Footnote, a helpful and nerdy study and exam prep assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say you don't know. Be clear, precise, and exam-oriented."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
  ]
  input_tokens = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
  )

  input_ids = input_tokens["input_ids"].to(llm.device)

  prompt_len = input_ids.shape[-1]

  generated_ids = llm.generate(
    input_ids,
    max_new_tokens=200
  )

  new_tokens = generated_ids[0][prompt_len:]

  generated_text = tokenizer.decode(
    new_tokens,
    skip_special_tokens=True
  ).strip()

  return generated_text

#public api:
def ask(question):
  context = retrieve(question)
  generated_ans = generate(context, question)
  return generated_ans, context