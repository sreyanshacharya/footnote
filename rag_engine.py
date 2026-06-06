import faiss
import os
import pickle
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
import time
import threading


#load embedding model
emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')

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
  device_map='auto',
  attn_implementation="sdpa"
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
# 1. change this back in your global setup to let accelerate handle 4-bit hooks cleanly:
# device_map="auto"

def generate(context, question):
  messages = [
    {"role": "system", "content" : "You are Footnote, a helpful and nerdy study and exam prep assistant. Answer the user's question using ONLY the provided context. If the answer is not in the context, say you don't know. Be clear, precise, and exam-oriented."},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
  ]
  
  # render the prompt as a raw string first
  prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
  )
  
  input_tokens = tokenizer(prompt_text, return_tensors="pt")
  
  input_ids = input_tokens["input_ids"].to(llm.device)
  attention_mask = input_tokens["attention_mask"].to(llm.device)

  streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

  generation_kwargs = dict(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=200,
    pad_token_id=tokenizer.eos_token_id,
    streamer=streamer
  )

  torch.cuda.reset_peak_memory_stats()
  start_vram = torch.cuda.memory_allocated()

  start_time = time.perf_counter()

  thread = threading.Thread(target=llm.generate, kwargs=generation_kwargs)
  thread.start()

  first_token_time = None
  token_count = 0
  generated_text = ""

  for chunk in streamer:
    if first_token_time is None:
      first_token_time = time.perf_counter()
      ttft = (first_token_time - start_time) * 1000
      print(f"\n[LOG] Time To First Token (TTFT) : {ttft:.4f} seconds")
      token_count += 1
    
    generated_text += (chunk)
    token_count += 1

  end_time = time.perf_counter()
  total_time_after_first = end_time - first_token_time #type: ignore
  itl = (total_time_after_first / max(1, token_count - 1)) * 1000

  end_vram = torch.cuda.memory_allocated()
  peak_vram = torch.cuda.max_memory_allocated()

  vram_delta_mb = (end_vram - start_vram) / (1024 ** 2)
  peak_vram_gb = peak_vram / (1024 ** 3)
  print(f"\n[LOG] Total Tokens Generated : {token_count}")
  print(f"\n[LOG] Inter-Token Latency (ITL) : {itl:.4f}ms/token")
  print(f"\n[LOG] VRAM Delta : {vram_delta_mb:+.4f} MB | Peak VRAM Usage : {peak_vram_gb:.4f} GB")

  return generated_text

#public api:
def ask(question):
  # profile retrieval
  start_time = time.time()
  context = retrieve(question)
  retrieval_time = time.time() - start_time
  
  # profile generation
  start_time = time.time()
  generated_ans = generate(context, question)
  generation_time = time.time() - start_time
  
  print(f"\n[profile] retrieval took: {retrieval_time:.4f} seconds")
  print(f"[profile] generation took: {generation_time:.4f} seconds")
  print(f"[profile] tokens per second: {200 / generation_time:.2f} t/s\n")

  torch.cuda.empty_cache()
  
  return generated_ans, context