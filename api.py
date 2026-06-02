from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from pathlib import Path
import shutil

from rag_engine import ask
from ingest import ingest

app = FastAPI()

class QueryInput(BaseModel):
  question : str

@app.get('/')
async def root():
  return {"message" : "hello bread, please go to /ask to ask the question related to current sample - COA textbook"}

@app.post('/ask')
def askapi(item : QueryInput):
  answ, context = ask(item.question)
  return{"answer" : answ,
         "context" : context}

@app.post('/ingest')
def ingapi(file : UploadFile = File(...)):
  data_dir = Path("data")
  data_dir.mkdir(exist_ok=True)
  save_path = f"{data_dir}/{file.filename}"

  with open(save_path, "wb") as f:
    shutil.copyfileobj(file.file, f)

  pdfcount, chunklength = ingest()
  return {
    "ingested no. of pdfs" : pdfcount,
    "length of chunks" : chunklength
    }