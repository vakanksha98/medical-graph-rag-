
#  Medical Graph RAG + Agentic AI

An AI-powered Medical Assistant built using Graph RAG, Vector Embeddings and Agentic AI.

##  What it does
- Takes patient symptoms as input
- Searches a Medical Knowledge Graph using Graph RAG
- Uses Vector Embeddings to understand meaning of symptoms
- Agentic Gemini AI reasons over results and gives structured medical report

##  Tech Stack
- **Google Gemini** — Agentic AI reasoning (Free API)
- **NetworkX** — Medical Knowledge Graph
- **Sentence Transformers** — Vector Embeddings (Real RAG)
- **Gradio** — Web Interface
- **PyPDF2** — PDF ingestion from WHO publications

##  Model Performance
| Test Type | Accuracy |
|---|---|
| Standard medical terms | 100% |
| Real world patient language | 20-40% |

## How to Run
1. Open `medical_graph_rag_FINAL.ipynb` in Google Colab
2. Get free API key from [aistudio.google.com](https://aistudio.google.com)
3. Paste key in Cell 2
4. Run all cells top to bottom

##  Key Concepts
| Concept | Implementation |
|---|---|
| Graph RAG | NetworkX knowledge graph traversal |
| Vector RAG | Sentence Transformers embeddings |
| Agentic AI | Gemini tool calling loop |
| Knowledge Base | WHO medical PDFs |

## Disclaimer
This is a research project. Always consult a real doctor for medical advice.
