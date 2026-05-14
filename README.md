# Medical Assistant — Graph RAG + Agentic AI
### Powered by Google Gemini (FREE!)

**What this project does:**
- Builds a **Knowledge Graph** of diseases, symptoms, medicines
- Uses **Graph RAG** + **Vector Embeddings** for smart search
- Uses **Agentic Gemini AI** to reason and give structured reports
- Supports **PDF upload** to expand the knowledge base
- Includes **Model Evaluation** to measure accuracy

**Tech Stack:** Python · NetworkX · Sentence Transformers · Google Gemini · Gradio

---

## Try the Live Demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akanksha/MedGraph-AI)

**Live Demo:** https://huggingface.co/spaces/akanksha/MedGraph-AI

---

## Features

- **🔍 Graph RAG Search** - Follow symptom → disease → medicine → side_effect connections
- **📊 Real Vector Embeddings** - Understands MEANING (finds "waking up to pee" → "Frequent Urination" → "Diabetes")
- **🤖 Agentic AI** - Gemini autonomously decides when to use keyword vs vector search
- **📄 PDF Upload** - Expand knowledge graph with WHO medical PDFs
- **📈 Model Evaluation** - Easy test (medical terms) vs Hard test (patient language)

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

### Configuration

Create a `.env` file with your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Or set it when prompted by the app.

## Usage

1. **Graph RAG Search** - Enter symptoms like "headache, dizziness, chest pain"
2. **Vector Search** - Enter natural language like "I keep waking up at night to pee"
3. **Medicine Info** - Ask about side effects of any medicine
4. **PDF Upload** - Add medical PDFs to expand the knowledge graph

## Project Structure

```
medical-graph-rag-/
├── medical_graph_rag_FINAL.ipynb  # Main notebook
├── app.py                          # Web app (Gradio)
├── requirements.txt               # Dependencies
├── SPEC.md                        # Technical documentation
└── README.md                      # This file
```

## Disclaimer

This is a **research project**. Always consult a qualified doctor for medical advice. The AI assistant is not a replacement for professional medical diagnosis.

## License

MIT License - See [LICENSE](LICENSE) for details.

## Author

**Akanksha Verma**
- M.Tech – Computer Science and Data Processing
- IIT Kharagpur