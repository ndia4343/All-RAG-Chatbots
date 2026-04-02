import streamlit as st
import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ShopBot AI | Smart Sales Concierge",
    page_icon="🤖",
    layout="wide",
)

# --- PROFESSIONAL DASHBOARD STYLING ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@700;800&display=swap" rel="stylesheet">
<style>
    .stApp { background-color: #f8fafc; color: #0f172a; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #0c1e3e; border-right: 1px solid #1e293b; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    .header-container { background-color: #1e3a8a; padding: 1.5rem; border-radius: 12px; color: white; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
    .header-title { font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 1.8rem; margin: 0; }
    .nav-label { font-size: 0.7rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 1.5rem; }
    .chat-bubble { padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem; max-width: 80%; }
    .bot-bubble { background: white; border: 1px solid #e2e8f0; border-left: 5px solid #2563eb; color: #1e293b; }
    .user-bubble { background: #2563eb; color: white; margin-left: auto; }
    .stats-card { background: #162447; padding: 1rem; border-radius: 8px; border: 1px solid #1f3a93; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# --- RAG ACCURACY & ANALYTICS ENGINE ---
@st.cache_resource
def load_assets(path):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    if not os.path.exists(path):
        data = {'Product':['Premium Wireless Headphones','Smart Fitness Watch','Organic Arabica Coffee','Mechanical Gaming Keyboard','Ergonomic Office Chair','UltraWide Monitor 34"','Stainless Steel Water Bottle'],
                'Category':['Electronics','Wearables','Grocery','Accessories','Furniture','Electronics','Home'],
                'Price':[249.99, 129.50, 18.0, 89.0, 350.0, 499.99, 25.0],
                'Stock Status':['In Stock','In Stock','Low Stock','Out of Stock','In Stock','In Stock','In Stock'],
                'Description':['Active noise cancellation and 30hr battery.','Heart rate and sleep tracking waterproof.','Single origin Ethiopian medium-roast.','RGB backlit blue-switch mechanical.','Adjustable lumbar and mesh back.','144Hz refresh HDR curved display.','Double-walled thermal insulation.']}
        df = pd.DataFrame(data)
        df.to_csv(path, index=False)
    _df = pd.read_csv(path)
    blobs = _df.apply(lambda r: f"Item: {r['Product']}, Price: ${r['Price']}, Details: {r['Description']}", axis=1).tolist()
    embs = embedder.encode(blobs)
    idx = faiss.IndexFlatL2(embs.shape[1])
    idx.add(np.array(embs).astype('float32'))
    stats = {"Total_Value": _df['Price'].sum(), "Avg_Price": _df['Price'].mean(), "Item_Count": len(_df)}
    return embedder, idx, _df, blobs, stats

# Initialize
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ecommerce_sales.csv")
embedder, index, df, doc_blobs, global_stats = load_assets(CSV_PATH)

# --- SIDEBAR & DASHBOARD ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3081/3081559.png", width=60)
    st.markdown("### ShopBot AI Admin")
    st.markdown(f""" <div class='stats-card'> <div style='font-size:0.75rem; color:#93c5fd;'>Total Inventory Value</div> <div style='font-size:1.1rem; font-weight:800;'>${global_stats['Total_Value']:,.2f}</div> </div> """, unsafe_allow_html=True)
    
    api_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") or st.text_input("Gemini API Key", type="password")
    
    st.markdown("<p class='nav-label'>MODEL ENGINE</p>", unsafe_allow_html=True)
    model_choice = st.selectbox("Preferred Engine:", ["gemini-1.5-flash", "gemini-1.0-pro"], help="Switch if you see 404 errors.")

# --- MAIN INTERFACE ---
st.markdown('<div class="header-container"><div class="header-title">ShopBot AI Concierge</div><div style="font-size:0.8rem;">🔌 Database Connected</div></div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"Greetings! I am ShopBot AI. I've indexed your full store directory. How can I help?"}]

for m in st.session_state.messages:
    cls = "bot-bubble" if m["role"]=="assistant" else "user-bubble"
    st.markdown(f'<div class="chat-bubble {cls}">{m["content"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Enter your query..."):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)
    
    with st.spinner("ShopBot AI thinking..."):
        q_emb = embedder.encode([prompt]).astype('float32')
        _, I = index.search(q_emb, k=5)
        context_matches = "\n".join([doc_blobs[i] for i in I[0]])
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # First try the user's preferred model
                try:
                    model = genai.GenerativeModel(model_choice)
                    sys_inst = f"Expert E-Commerce Data Analyst. Inventory Value: ${global_stats['Total_Value']}. Context: {context_matches}"
                    response = model.generate_content(f"{sys_inst}\n\nUSER: {prompt}")
                    ans = response.text
                except Exception as inner_e:
                    # Fallback to stable version if Flash 1.5 404s
                    fallback_model = "gemini-1.0-pro"
                    model = genai.GenerativeModel(fallback_model)
                    response = model.generate_content(f"CONTEXT: {context_matches} | USER: {prompt}")
                    ans = response.text + f"\n\n*(Note: Fallback to {fallback_model} used due to API availability)*"
            except Exception as e:
                ans = f"⚠️ Recovery Error: {e}"
        else:
            ans = f"Search Context Found:\n{context_matches}\n\n(Tip: Enter API Key for full AI reasoning!)"
            
    st.markdown(f'<div class="chat-bubble bot-bubble">{ans}</div>', unsafe_allow_html=True)
    st.session_state.messages.append({"role":"assistant", "content":ans})
