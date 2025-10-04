# 🩺 Medibot: Retrieval-Augmented Clinical Reference Chatbot

**Medibot** is a context-aware Q&A assistant for medical reference, powered by a local PDF knowledge base and LLMs like Llama 4 Maverick (Groq) or Mistral 7B Instruct (HuggingFace). It leverages a FAISS vector store and sentence embeddings to ground answers in your curated documents—no hallucinations, just traceable, reference-backed responses!

---

## ✨ Features

- **Semantic PDF Search:** Fast, accurate retrieval using FAISS and SentenceTransformers.
- **Flexible LLM Backends:** Choose between Groq-hosted (Llama 4 Maverick) or HuggingFace (Mistral).
- **Source Traceability:** Every answer shows which document/chunk it came from.
- **CLI & UI:** Command-line prototype and Streamlit web chat.
- **Prompt Customization:** Easily adjust answer tone and style.

---

## 🏗️ Architecture

```
PDF(s) ──► Text Splitter ──► Embeddings ──► FAISS Vectorstore (vectorstore/db_faiss)
                                                 ▲
User Query ──► Retriever (top-k) ────────────────┘
                      │
              Prompt Assembly
                      │
          LLM Generation (Groq or HF)
                      │
          Answer + Source Chunks
```

---

## 🛠️ Components

| File                       | Role                                               |
|----------------------------|----------------------------------------------------|
| `create_memory_for_llm.py` | Build FAISS index from PDFs                        |
| `connect_memory_with_llm.py` | CLI Q&A using HuggingFace endpoint              |
| `medibot.py`               | Streamlit chat UI using Groq + FAISS retrieval     |
| `vectorstore/db_faiss`     | Persisted FAISS index (auto-generated)             |
| `data/`                    | PDF source documents                               |

---

## ⚙️ Setup & Installation

**1. Clone & Create Environment**
```bash
git clone <your-repo>
cd <your-repo>
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
# or, if using uv:
uv sync
```

**3. Set Environment Variables**
- Add to `.env` or export in your shell:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx
GROQ_API_KEY=groq_xxxxxxxxxxxxxxxxxxx
# Optional:
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HUGGINGFACE_REPO_ID=mistralai/Mistral-7B-Instruct-v0.3
```

---

## 🗂️ Build the Vector Store

Place your PDFs in `data/`. Then run:
```bash
python create_memory_for_llm.py
```
This will:
- Load PDFs from `data/`
- Chunk text with overlap
- Embed with SentenceTransformers
- Save the FAISS index to `vectorstore/db_faiss`

**Sample pipeline code:**
```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/your_doc.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, emb)
db.save_local("vectorstore/db_faiss")
```

---

## 💬 Usage

### Command-Line

```bash
source .venv/bin/activate
export HF_TOKEN=...  # if not in .env
python connect_memory_with_llm.py
# Enter your question, e.g.:
# How is hypertension managed?
```

### Streamlit Chat UI

```bash
source .venv/bin/activate
export GROQ_API_KEY=...
streamlit run medibot.py
# Open the shown URL (default: http://localhost:8501)
```

---

## 🔄 Embedding Modes

- **Local**: `get_vectorstore()`
- **Remote (HuggingFace API)**: `get_vectorstore_hf_api(token)`
    - Use if running on limited hardware or disk.

---

## 🛠️ Prompt Customization

Edit `CUSTOM_PROMPT_TEMPLATE` in either script to adjust style or tone.  
**Variables:** `{context}` and `{question}` must remain.

---

## 🧪 Quick Test

After building the vector store:
```bash
python - <<'PY'
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
emb = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db = FAISS.load_local('vectorstore/db_faiss', emb, allow_dangerous_deserialization=True)
print('Index loaded. k=2 sample:\n', db.similarity_search('What is diabetes?', k=2))
PY
```

---

## 🔍 Source Trace Display

Both scripts return `source_documents`. To display snippets:
```python
for i, d in enumerate(source_documents, 1):
    snippet = d.page_content[:300].replace('\n', ' ')
    print(f"[Source {i}] {snippet}...")
```

---

## 🐛 Troubleshooting

| Symptom                                   | Cause                          | Fix                                          |
|--------------------------------------------|--------------------------------|----------------------------------------------|
| `text_generation() unexpected keyword`     | Wrong arg to HF API            | Use `huggingfacehub_api_token` (see docs)    |
| `FAISS.load_local ... file not found`      | Vectorstore not built          | Run `create_memory_for_llm.py` first         |
| Empty/irrelevant answers                   | Small `k`, chunk mismatch      | Increase `search_kwargs={'k':5}` or rebuild  |
| Hallucinations                            | LLM ignoring context           | Tighten prompt, lower temp, reduce tokens    |
| `HF_TOKEN not set` error                   | Missing token                  | Export or add to `.env`                      |
| Virtualenv mismatch warning                | Old VIRTUAL_ENV variable       | `deactivate` then `source .venv/bin/activate`|

---

## 🧱 Extending

- Support multiple PDFs (glob `data/*.pdf`)
- Enable streaming tokens in UI
- Add OpenAI/Anthropic backend
- Persist chat history with sources
- Evaluation harness (e.g., RAGAS)

---

## ⚖️ Disclaimer

> This tool is for educational and reference purposes only.  
> **It does not provide medical advice, diagnosis, or treatment.**  
> Always consult a licensed healthcare professional.

---

## ✅ Quick Recap

```bash
python create_memory_for_llm.py         # Build index (once)
streamlit run medibot.py                # Chat UI (needs GROQ_API_KEY)
python connect_memory_with_llm.py       # CLI (needs HF_TOKEN)
```
