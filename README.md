# 📘 AI RAG Course Assistant (Gemini + FAISS + FastAPI + Streamlit)

An educational Retrieval-Augmented Generation (RAG) app:
- Upload your **lecture PDFs** (or a ZIP of PDFs).
- Ask questions in natural language.
- The app retrieves relevant chunks (FAISS) and answers using **Google Gemini** via LangChain.

## 🧰 Tech
- **Embeddings:** `GoogleGenerativeAIEmbeddings` (`models/embedding-001`)
- **LLM:** `ChatGoogleGenerativeAI` (default `gemini-pro`, configurable)
- **Vector DB:** FAISS (persisted locally)
- **Backend:** FastAPI
- **Frontend:** Streamlit

---

## ⚙️ Setup

### 1) Clone & enter the project
```bash
git clone <your-repo-url>
cd AI-Course-Assistant-Gemini-FAISS
```

### 2) Create a Google API Key
- Get a key for **Google Generative AI** (Gemini).
- Set it in an `.env` file in **backend/** as:
  ```env
  GOOGLE_API_KEY=your_key_here
  # Optional overrides:
  GEMINI_MODEL_NAME=gemini-pro
  CHUNK_SIZE=1200
  CHUNK_OVERLAP=200
  TOP_K=4
  TEMPERATURE=0.2
  DATA_DIR=data
  PERSIST_DIR=vectorstore/faiss_index
  ```

---

## ▶️ Run

### Backend
```bash
cd backend
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python app.py
```
- Backend runs at `http://127.0.0.1:8000`

### Frontend
```bash
cd ../frontend
pip install -r requirements.txt
streamlit run app.py
```
- Frontend runs at `http://localhost:8501`

---

## 🧪 Use
1. Open the Streamlit page.
2. (Optional) Change **Backend URL** in the sidebar.
3. Upload one or more **PDFs** or a **ZIP** that contains PDFs.
4. Ask questions. The assistant will cite sources (file + page).

---

## 🛠️ API Endpoints (FastAPI)
- `GET /health` → health check
- `GET /stats` → pipeline stats
- `POST /reset` → clear FAISS index
- `POST /params` (form fields: `k`, `temperature`) → update retrieval/LLM params
- `POST /upload` (multipart, `file`) → upload a PDF or a ZIP of PDFs
- `POST /ask` (JSON `{question, k?, temperature?}`) → get answer + sources

---

## ❓ FAQ
- **Where is the vector DB stored?** `vectorstore/faiss_index/` (configurable).
- **Can I change the model?** Set `GEMINI_MODEL_NAME` in `.env` (e.g., `gemini-1.5-flash`).
- **Do I need OpenAI?** No. Only **Google Gemini** is used here.
