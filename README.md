# All-RAG-Chatbots 🤖💎

A premium collection of high-performance **Retrieval-Augmented Generation (RAG)** chatbots, designed for portfolio-grade demonstrations and real-world utility. This repository showcases advanced AI architectures using local embeddings with FAISS and state-of-the-art LLMs.

## 🚀 Projects Included

### 1. [1Mart-RAG-Bot](./1Mart-RAG-Bot)
An intelligent E-Commerce Sales & Support Assistant.
- **Goal:** Help users find products, check stock, and get detailed recommendations.
- **Tech Stack:** Streamlit, FAISS, Sentence-Transformers, Google Gemini Flash API.
- **Key Feature:** Local vector search for near-instant product lookup even with large catalogs.

### 2. [Indigo-CarPrices-Bot](./Indigo-CarPrices-Bot)
A precision Data Analysis Explorer for the UCI Car Prices Dataset.
- **Goal:** Provide detailed analytical insights and price predictions based on historical data.
- **Tech Stack:** Streamlit, Pandas, Google Gemini API, Custom CSS Branding.
- **Key Feature:** High-fidelity data ingestion and professional reporting format.

---

## 🛠️ Global Requirements

To run any of the chatbots, ensure you have:
- Python 3.9+
- A Google Gemini API Key (from [Google AI Studio](https://aistudio.google.com/))
- The following common libraries:
  ```bash
  pip install streamlit pandas sentence-transformers faiss-cpu google-generativeai
  ```

## 📂 Repository Structure

```text
.
├── 1Mart-RAG-Bot/          # E-Commerce Support Chatbot
│   ├── app.py              # Main Application
│   ├── requirements.txt    # Project Dependencies
│   └── ecommerce_sales.csv  # Sample Product Database
├── Indigo-CarPrices-Bot/    # Data Analysis Engine
│   ├── app.py              # Streamlit Interface
│   ├── flask_app.py       # API Backend
│   └── ...
└── README.md               # Landing Page & Overview
```

## ✨ Created with Passion
Designed for US clients and high-end portfolio presentations. Clean code, professional UI, and robust RAG logic.
