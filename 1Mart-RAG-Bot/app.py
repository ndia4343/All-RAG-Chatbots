import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import requests
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="1Mart | AI Commerce Assistant",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- THEME & STYLING ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@600;800&display=swap" rel="stylesheet">
<style>
    :root {
        --primary: #10b981;
        --primary-dark: #059669;
        --bg-dark: #0f172a;
        --surface-dark: #1e293b;
        --text-muted: #94a3b8;
    }
    
    .stApp {
        background-color: var(--bg-dark);
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--surface-dark);
        border-right: 1px solid #334155;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #34d399 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Chat bubbles */
    .chat-bubble {
        padding: 1.5rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #334155;
        max-width: 85%;
    }
    
    .user-bubble {
        background: #334155;
        align-self: flex-end;
        margin-left: auto;
    }
    
    .bot-bubble {
        background: #1e293b;
        border-left: 4px solid var(--primary);
    }
    
    /* Product card style result */
    .product-card {
        background: rgba(16, 185, 129, 0.05);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 0.5rem;
    }
    
    .status-badge {
        font-size: 0.7rem;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        background: var(--primary-dark);
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- LOGIC & RAG ENGINE ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_and_process_data(csv_path):
    if not os.path.exists(csv_path):
        # Create sample data if it doesn't exist
        data = {
            'Product': ['Premium Wireless Headphones', 'Smart Fitness Watch', 'Organic Arabica Coffee', 'Mechanical Gaming Keyboard', 'Ergonomic Office Chair', 'UltraWide Monitor 34"', 'Stainless Steel Water Bottle'],
            'Category': ['Electronics', 'Wearables', 'Grocery', 'Accessories', 'Furniture', 'Electronics', 'Home'],
            'Price': [249.99, 129.50, 18.00, 89.00, 350.00, 499.99, 25.00],
            'Stock Status': ['In Stock', 'In Stock', 'Low Stock', 'Out of Stock', 'In Stock', 'In Stock', 'In Stock'],
            'Description': [
                'Noise canceling over-ear headphones with 30h battery life and high fidelity sound.',
                'Tracks heart rate, steps, and sleep with OLED display and waterproof design.',
                '100% organic beans sourced from Ethiopia, medium roast with notes of chocolate.',
                'RGB backlit mechanical keyboard with blue switches and durable aluminum frame.',
                'Adjustable lumbar support, breathable mesh back, and 4D armrests for long work sessions.',
                'Immersive 34-inch curved display with 144Hz refresh rate and HDR10 support.',
                'Double-wall insulated bottle keeps drinks cold for 24h or hot for 12h.'
            ]
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return df
    return pd.read_csv(csv_path)

@st.cache_resource
def build_vector_store(_df):
    model = load_embedding_model()
    # Combine columns into a searchable blob
    texts = _df.apply(lambda r: f"Product: {r['Product']} | Category: {r['Category']} | Price: ${r['Price']} | Status: {r['Stock Status']} | Description: {r['Description']}", axis=1).tolist()
    embeddings = model.encode(texts)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, texts

# Initialize
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ecommerce_sales.csv")

df = load_and_process_data(CSV_PATH)
index, text_documents = build_vector_store(df)
embedding_model = load_embedding_model()

# --- SIDEBAR & API CONFIG ---
with st.sidebar:
    st.markdown("## 🛒 1Mart Control Panel")
    st.info("E-Commerce Intelligence Engine powered by RAG.")
    
    # Try to find API key in secrets or env
    env_api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if env_api_key:
        api_key = env_api_key
        st.success("✅ API Key loaded from system secrets.")
    else:
        api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google AI Studio API key for high-quality responses.")
        if not api_key:
            st.warning("⚠️ Enter API key to enable AI reasoning.")
    
    st.markdown("---")
    st.markdown("### Inventory Stats")
    st.metric("Total Products", len(df))
    st.metric("Categories", len(df['Category'].unique()))
    
    if st.button("Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# --- CHAT INTERFACE ---
st.markdown('<div class="main-title">1Mart AI Concierge</div>', unsafe_allow_html=True)
st.caption("Ask me about our products, stock status, or recommendations.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    role_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
    st.markdown(f'<div class="chat-bubble {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("What are you looking for today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)
    
    with st.spinner("Analyzing inventory..."):
        # 1. Similarity Search
        query_vector = embedding_model.encode([prompt]).astype('float32')
        D, I = index.search(query_vector, k=3)
        
        context = "\n".join([text_documents[idx] for idx in I[0]])
        
        # 2. Generation
        if api_key:
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                
                system_prompt = f"""
                You are a professional e-commerce customer support bot for '1Mart'. 
                Answer the user's question using the provided context. 
                If the product isn't in the context, be polite and say we don't have it.
                ALWAYS format prices clearly.
                
                Context:
                {context}
                """
                
                response = model.generate_content(prompt + "\n\n(Context provided above)")
                full_response = response.text
                
            except Exception as e:
                full_response = f"I found some relevant items, but I'm having trouble generating a detailed response. Error: {str(e)}\n\n**Quick Matches:**\n" + context
        else:
            # Simple fallback if no API key
            full_response = "I found the following items that might interest you:\n\n" + context + "\n\n*(Pro-tip: Provide a Gemini API key in the sidebar for a better conversational experience)*"
            
    st.markdown(f'<div class="chat-bubble bot-bubble">{full_response}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
