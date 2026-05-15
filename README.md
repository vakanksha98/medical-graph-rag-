---
title: MedGraph AI
emoji: ⚕
colorFrom: green
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
python_version: 3.10
app_file: app.py
pinned: false
---

# Medical Assistant — Graph RAG + Agentic AI
### Powered by Google Gemini (FREE!)

**What this project does:**
- Builds a **Knowledge Graph** of diseases, symptoms, medicines
- Uses **Graph RAG** + **Vector Embeddings** for smart search
- Uses **Agentic Gemini AI** to reason and give structured reports
- Supports **PDF upload** to expand the knowledge base

**Tech Stack:** Python · NetworkX · Sentence Transformers · Google Gemini · Gradio

---

## Features

- **🔍 Graph RAG Search** - Follow symptom → disease → medicine → side_effect connections
- **📊 Real Vector Embeddings** - Understands MEANING (finds "waking up to pee" → "Frequent Urination" → "Diabetes")
- **🤖 Agentic AI** - Gemini autonomously decides when to use keyword vs vector search
- **📄 PDF Upload** - Add medical PDFs to expand the knowledge graph

## How It Works

```
Patient Symptoms
       ↓
Vector Embedding Search (Sentence Transformers)
       ↓
Knowledge Graph Traversal (NetworkX)
       ↓
Gemini Agentic Reasoning
       ↓
Structured Medical Report
```

## Model Performance

| Test Type | Accuracy |
|-----------|----------|
| Standard medical terms | 100% |
| Real world patient language | 20-40% |

## Quick Start (Local)

### Prerequisites

- Python 3.8+
- Google Gemini API Key (FREE at https://aistudio.google.com/)

### Installation

```bash
# Clone repository
git clone https://github.com/vakanksha98/medical-graph-rag-.git
cd medical-graph-rag-

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

## Disclaimer

This is a **research project**. Always consult a qualified doctor for medical advice. The AI assistant is not a replacement for professional medical diagnosis.

## Author

**Akanksha Verma**
- M.Tech – Computer Science and Data Processing
- IIT Kharagpur