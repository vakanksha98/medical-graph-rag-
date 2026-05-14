# Medical Graph RAG + Agentic AI

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python)
![Google Gemini](https://img.shields.io/badge/Gemini-API-4285F4?logo=google)
![NetworkX](https://img.shields.io/badge/NetworkX-3.0+-DC3535?logo=networkx)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF6B6B?logo=gradio)
![License](https://img.shields.io/badge/License-MIT-green)

An AI-powered Medical Assistant built using Graph RAG, Vector Embeddings, and Agentic AI that provides structured medical reports based on patient symptoms.

## Overview

This project combines knowledge graph traversal with vector semantic search and agentic reasoning to create a medical diagnosis assistance system. It processes patient symptoms through multiple AI layers to generate comprehensive medical reports.

## Features

- **Graph RAG** - Knowledge graph traversal for structured medical reasoning
- **Vector RAG** - Semantic embedding search for understanding layperson terms
- **Agentic AI** - Multi-turn reasoning with Google Gemini
- **Medical Knowledge Base** - WHO publication PDFs
- **Gradio Interface** - User-friendly web UI

## Tech Stack

| Category | Technology |
|----------|------------|
| AI Reasoning | Google Gemini API |
| Knowledge Graph | NetworkX |
| Vector Embeddings | Sentence Transformers |
| Web Interface | Gradio |
| PDF Processing | PyPDF2 |
| Environment | Jupyter Notebook |

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

## Project Structure

```
medical-graph-rag-/
├── medical_graph_rag_FINAL.ipynb  # Main notebook
├── README.md                       # This file
└── requirements.txt               # Dependencies
```

## Installation

```bash
# Clone repository
git clone https://github.com/vakanksha98/medical-graph-rag-.git
cd medical-graph-rag-

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

```
google-generativeai>=0.3.0
networkx>=3.0
sentence-transformers>=2.2.0
gradio>=4.0.0
pypdf2>=3.0.0
torch>=2.0.0
transformers>=4.30.0
```

## Usage

1. Open `medical_graph_rag_FINAL.ipynb` in Google Colab
2. Get free API key from [aistudio.google.com](https://aistudio.google.com)
3. Paste key in Cell 2
4. Run all cells sequentially
5. Access the Gradio interface

## Key Concepts

| Concept | Implementation |
|---------|---------------|
| Graph RAG | NetworkX knowledge graph traversal |
| Vector RAG | Sentence Transformers embeddings |
| Agentic AI | Gemini tool calling loop |
| Knowledge Base | WHO medical PDFs |

## Disclaimer

This is a **research project**. Always consult a qualified doctor for medical advice. The AI assistant is not a replacement for professional medical diagnosis.

## Portfolio Presentation

This project demonstrates:
- RAG (Retrieval Augmented Generation) architecture
- Knowledge graph construction and traversal
- Vector embeddings for semantic search
- Agentic AI with tool-use capabilities
- Healthcare AI application development

## Future Improvements

- [ ] Improve real-world patient language understanding
- [ ] Expand medical knowledge graph
- [ ] Add more PDF sources
- [ ] Implement multi-language support
- [ ] Add symptom severity assessment
- [ ] Deploy as web service

## Author

**Akanksha Verma**
- M.Tech – Computer Science and Data Processing
- IIT Kharagpur

## License

MIT License - See [LICENSE](LICENSE) for details.