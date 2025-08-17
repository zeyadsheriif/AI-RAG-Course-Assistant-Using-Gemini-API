import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag_pipeline import RAGPipeline

import uvicorn 

load_dotenv()

app = FastAPI(title="AI Course Assistant (Gemini + FAISS)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)


rag = RAGPipeline(
    persist_dir=os.getenv("PERSIST_DIR", "vectorstore/faiss_index"),
    chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
    k=int(os.getenv("TOP_K", "4")),
    temperature=float(os.getenv("TEMPERATURE", "0.2")),
    model_name=os.getenv("GEMINI_MODEL_NAME", None)
)

class AskPayload(BaseModel):
    question: str
    k: Optional[int] = None
    temperature: Optional[float] = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/stats")
async def stats():
    return rag.stats()

@app.post("/reset")
async def reset():
    rag.reset_index()
    return {"message": "Vector store cleared."}

@app.post("/params")
async def update_params(k: Optional[int] = Form(None), temperature: Optional[float] = Form(None)):
    updated = rag.set_params(k=k, temperature=temperature)
    return {"updated": updated}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    dest = DATA_DIR / file.filename
    with open(dest, "wb") as f:
        f.write(await file.read())

    summary = rag.add_file(str(dest))
    return {"message": "File(s) ingested.", **summary}

@app.post("/ask")
async def ask(payload: AskPayload):
    if payload.k is not None or payload.temperature is not None:
        rag.set_params(k=payload.k, temperature=payload.temperature)
    res = rag.ask(payload.question)
    return res



if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
