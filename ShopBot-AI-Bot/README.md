# 1Mart-RAG-Bot 🛒🚀

The **1Mart-RAG-Bot** is an intelligent E-Commerce Sales & Support Assistant that uses a **Retrieval-Augmented Generation (RAG)** architecture. It's designed to provide accurate answers about products, prices, and stock status using a local vector database.

## 🌟 Key Features

- **High-Speed Search:** Uses **FAISS** index for millisecond search over product catalogs.
- **Natural Language Understanding:** Powered by **Sentence-Transformers** for semantic search.
- **Professional Replies:** Optimized for **Google Gemini 1.5 Flash** for high-quality, conversational responses.
- **Premium UI:** Glassmorphism-inspired Streamlit interface with emerald-accented branding.
- **Portfolio-Ready:** Ready to be deployed and presented to clients as a production-grade AI agent.

## 🛠️ Technical Stack

- **Framework:** Streamlit
- **Embeddings:** `all-MiniLM-L6-v2` (Local)
- **Vector Index:** FAISS
- **LLM:** Google Gemini 1.5 Flash (API-based)
- **Data Handling:** Pandas & NumPy

## 📦 Requirements

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## 🚀 How to Run Locally

1. Clone this repository.
2. Navigate to the project directory:
   ```bash
   cd 1Mart-RAG-Bot
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
4. Enter your **Google Gemini API Key** in the sidebar to enable full conversational capabilities.

## 📝 Data Source
The application includes a sample `ecommerce_sales.csv` for demonstration. You can replace this with your own product catalog to customize the bot for any e-commerce business.

## ✨ Portfolio Impression
This project demonstrates the ability to combine local search efficiency with the power of modern LLMs, creating a responsive and reliable e-commerce experience.
