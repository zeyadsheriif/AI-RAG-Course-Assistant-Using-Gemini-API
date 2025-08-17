
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

class RAGPipeline:
    """
    A simple RAG pipeline using:
    - Embeddings: GoogleGenerativeAIEmbeddings ("models/embedding-001")
    - LLM: ChatGoogleGenerativeAI (default "gemini-pro")
    - Vector store: FAISS (saved/loaded locally)
    """

    def __init__(
        self,
        persist_dir: str = "vectorstore/faiss_index",
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        k: int = 4,
        temperature: float = 0.2,
        model_name: str = None
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k = k
        self.temperature = temperature

        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-pro")

        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=self.temperature)

        self.db = None
        if any(self.persist_dir.iterdir()):
            self._load_index()

        self.qa_chain = None
        if self.db is not None:
            self._build_qa_chain()

    def _build_qa_chain(self) -> None:
        retriever = self.db.as_retriever(search_kwargs={"k": self.k})

        template = (
            "You are an **AI Course Assistant**. Answer the user's question using only the provided context.\n"
            "If the answer cannot be found in the context, say you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:\n"
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template,
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def _save_index(self) -> None:
        if self.db is not None:
            self.db.save_local(str(self.persist_dir))

    def _load_index(self) -> None:
        self.db = FAISS.load_local(
            str(self.persist_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def _ensure_index(self) -> None:
        if self.db is None:
            raise RuntimeError("Vector index is empty. Upload PDFs first.")


    def set_params(self, k: int = None, temperature: float = None) -> Dict[str, Any]:
        updated = {}
        if k is not None:
            self.k = int(k)
            updated["k"] = self.k
        if temperature is not None:
            self.temperature = float(temperature)
            self.llm = ChatGoogleGenerativeAI(model=self.model_name, temperature=self.temperature)
            updated["temperature"] = self.temperature

        if self.db is not None:
            self._build_qa_chain()

        return updated

    def reset_index(self) -> None:
        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.db = None
        self.qa_chain = None

    def add_file(self, file_path: str) -> Dict[str, Any]:
        """
        Add a single PDF, or extract a ZIP and add all PDFs inside it.
        Returns summary dict with counts.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")

        pdf_paths: List[Path] = []

        if path.suffix.lower() == ".zip":
            extract_dir = path.with_suffix("")
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(extract_dir)
            for p in extract_dir.rglob("*.pdf"):
                pdf_paths.append(p)
        elif path.suffix.lower() == ".pdf":
            pdf_paths.append(path)
        else:
            raise ValueError("Only .pdf or .zip files are supported.")

        docs = []
        for pdf in pdf_paths:
            loader = PyPDFLoader(str(pdf))
            docs.extend(loader.load())

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)

        if self.db is None:
            self.db = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.db.add_documents(chunks)

        self._save_index()
        self._build_qa_chain()

        return {
            "files_added": [str(p) for p in pdf_paths],
            "num_pages": len(docs),
            "num_chunks": len(chunks),
        }

    def ask(self, question: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            return {
                "answer": "No documents available. Please upload lecture notes first.",
                "sources": []
            }
        res = self.qa_chain({"query": question})
        answer = res.get("result", "")
        src_docs = res.get("source_documents", []) or []

        sources = []
        for d in src_docs:
            meta = d.metadata or {}
            sources.append({
                "source": meta.get("source"),
                "page": (meta.get("page", 0) + 1) if isinstance(meta.get("page"), int) else meta.get("page")
            })

        return {"answer": answer, "sources": sources}

    def stats(self) -> Dict[str, Any]:

        has_index = self.db is not None
        return {
            "has_index": has_index,
            "persist_dir": str(self.persist_dir),
            "k": self.k,
            "temperature": self.temperature,
            "model_name": self.model_name
        }
