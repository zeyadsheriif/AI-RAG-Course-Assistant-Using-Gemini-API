
import io
import requests
import streamlit as st

st.set_page_config(page_title="📘 AI Course Assistant", layout="wide")
st.title("📘 AI Course Assistant (Gemini + FAISS)")


with st.sidebar:
    st.header("Settings")
    backend_url = st.text_input("Backend URL", value="http://127.0.0.1:8000")
    top_k = st.slider("Top-K Context Chunks", 1, 10, 4)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.1)

    st.divider()
    if st.button("Reset Vector Store"):
        try:
            r = requests.post(f"{backend_url}/reset")
            st.success(r.json().get("message", "Reset."))
        except Exception as e:
            st.error(f"Reset failed: {e}")

    st.divider()
    st.caption("Tip: upload PDFs or a ZIP containing PDFs.")


st.subheader("📤 Upload Lecture Notes")
uploads = st.file_uploader("Upload PDF(s) or a ZIP", type=["pdf", "zip"], accept_multiple_files=True)
if uploads:
    for up in uploads:
        try:
            file_bytes = up.read()
            files = {"file": (up.name, io.BytesIO(file_bytes), "application/octet-stream")}
            r = requests.post(f"{backend_url}/upload", files=files, timeout=120)
            if r.status_code == 200:
                msg = r.json().get("message", "Uploaded.")
                pages = r.json().get("num_pages", "?")
                chunks = r.json().get("num_chunks", "?")
                st.success(f"✅ {up.name}: {msg} (pages: {pages}, chunks: {chunks})")
            else:
                st.error(f"❌ {up.name}: {r.status_code} {r.text}")
        except Exception as e:
            st.error(f"❌ {up.name}: {e}")

st.divider()

if "chat" not in st.session_state:
    st.session_state.chat = []  

st.subheader("💬 Ask a Question")
user_q = st.chat_input("Type your question...")

cols = st.columns([2, 1])
with cols[0]:
    if user_q:
        st.session_state.chat.append({"role": "user", "content": user_q})
        try:
            payload = {"question": user_q, "k": top_k, "temperature": temperature}
            r = requests.post(f"{backend_url}/ask", json=payload, timeout=120)
            if r.status_code == 200:
                data = r.json()
                answer = data.get("answer", "")
                sources = data.get("sources", [])
                st.session_state.chat.append({"role": "assistant", "content": answer, "sources": sources})
            else:
                st.session_state.chat.append({"role": "assistant", "content": f"Error: {r.status_code} {r.text}", "sources": []})
        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"Request failed: {e}", "sources": []})


    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                sources = msg.get("sources") or []
                if sources:
                    with st.expander("Sources"):
                        for i, s in enumerate(sources, start=1):
                            src = s.get("source", "unknown")
                            page = s.get("page", "?")
                            st.write(f"{i}. `{src}` - page {page}")

with cols[1]:
    st.subheader("ℹ️ Status")
    try:
        s = requests.get(f"{backend_url}/stats", timeout=10).json()
        st.json(s)
    except Exception as e:
        st.error(f"Stats unavailable: {e}")
