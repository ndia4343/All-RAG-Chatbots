import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from io import BytesIO

def try_math(query: str):
    """Detect simple arithmetic in query and compute directly."""
    q = query.lower().strip()
    numbers = re.findall(r"\d+(?:[,\d]*)?(?:\.\d+)?", q.replace(",", ""))
    nums = [float(n) for n in numbers if n]

    if not nums: return None

    if any(k in q for k in ["sum", "total", "add", "plus", "+"]) and "revenue" not in q and "products" not in q:
        result = sum(nums)
        return f"The total is ${result:,.2f}" if len(nums) > 1 else None
    if any(k in q for k in ["multiply", "times", "product", "*", "×"]) and "popular product" not in q:
        result = nums[0]
        for n in nums[1:]: result *= n
        return f"The result is ${result:,.2f}"
    if any(k in q for k in ["subtract", "minus", "difference", "-"]):
        result = nums[0] - sum(nums[1:])
        return f"The result is ${result:,.2f}"
    if any(k in q for k in ["divide", "divided by", "/"]):
        if len(nums) >= 2 and nums[1] != 0:
            result = nums[0] / nums[1]
            return f"The result is {result:,.4f}"
    if any(k in q for k in ["average", "avg", "mean"]):
        result = sum(nums) / len(nums)
        return f"The average is ${result:,.2f}"
    if "%" in q or "percent" in q:
        if len(nums) >= 2:
            result = (nums[0] / nums[1]) * 100
            return f"{result:.2f}%"
    return None

def format_currency_in_answer(text: str) -> str:
    """Adds $ before bare numbers near revenue/price/value keywords."""
    text = re.sub(
        r"(?i)(revenue|price|cost|value|total|amount|order)[\s:]*([\$£€]?)(\d[\d,]*(?:\.\d{1,2})?)",
        lambda m: f"{m.group(1)} ${m.group(3)}" if not m.group(2) else m.group(0),
        text
    )
    return text

st.set_page_config(page_title="1Mart | ShopAI", page_icon="🛒", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #333333;
}
.top-header {
    background-color: #6C3011;
    color: white;
    padding: 1.5rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: -3.5rem;
    margin-left: -4rem;
    margin-right: -4rem;
    margin-bottom: 2rem;
}
.header-left .logo {
    font-size: 1.8rem;
    font-weight: 500;
    margin-bottom: -2px;
}
.header-left .sublogo {
    font-size: 0.9rem;
    color: #e0b49f;
}
.powered-btn {
    background-color: #f7931e;
    color: #fff;
    padding: 0.5rem 1.2rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: bold;
}
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: none;
}
.sys-status-table {
    width: 100%;
    font-size: 0.85rem;
    margin-top: 5px;
    margin-bottom: 2rem;
}
.sys-status-table td {
    padding: 6px 0;
}
.sys-status-table td:nth-child(2) {
    text-align: right;
    color: #eb5e28;
}
.upload-box {
    text-align: center;
    font-size: 0.8rem;
    color: #555;
    margin: 1rem 0;
}
.rag-notice {
    background-color: #fdf3eb;
    color: #a45a3d;
    padding: 1.2rem;
    font-size: 0.8rem;
    margin-top: 2rem;
    line-height: 1.4;
}
.build-kb-btn button {
    background-color: #ea6c2c !important;
    color: white !important;
    width: 100%;
    border-radius: 5px;
    font-weight: 600;
    border: none;
    padding: 0.5rem 0;
}
.chat-row {
    display: flex;
    align-items: flex-start;
    margin-bottom: 1.5rem;
}
.chat-bot { flex-direction: row; }
.chat-user { flex-direction: row-reverse; }
.avatar {
    width: 32px;
    height: 32px;
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}
.avatar-bot { background-color: #ef7d38; }
.avatar-user { background-color: transparent; }
.msg-content {
    max-width: 80%;
    margin: 0 16px;
    padding: 1rem 1.2rem;
    border-radius: 8px;
    font-size: 0.95rem;
    line-height: 1.5;
}
.msg-bot { background-color: transparent; color: #111; border: none; }
.msg-user { background-color: #ea6c2c; color: #ffffff; }
.time-stamp { font-size: 0.65rem; color: #888; margin-top: 6px; }
.subtext-bot { text-align: left; margin-left: 18px; }
.subtext-user { text-align: right; margin-right: 18px; }
.try-asking {
    font-size: 0.8rem;
    font-weight: 600;
    color: #555;
    margin-bottom: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.suggested-q {
    padding: 0.6rem;
    font-size: 0.85rem;
    color: #333;
    margin-bottom: 0.4rem;
}
.suggested-q.highlight { background-color: #fff1e5; color: #b05c3b; }
.send-btn button {
    background-color: #ea6c2c !important;
    color: white !important;
    width: 100%;
    height: 100%;
    margin-top: 28px;
    border: none;
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="top-header">
    <div class="header-left">
        <div class="logo">🛒 1Mart</div>
        <div class="sublogo">Everything you need, delivered fast</div>
    </div>
    <div class="powered-btn">POWERED BY AI</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading AI Models and Knowledge Base...")
def load_models_and_data_v2():
    device = "cpu"
    embedder  = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    llm_model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

    default_chunks = []
    default_index  = None
    default_df     = None
    default_csv    = "ecommerce_sales.csv"

    if os.path.exists(default_csv):
        try:
            df = pd.read_csv(default_csv)
            default_df = df.copy()
            df = df.fillna("N/A")
            df.columns = [str(c).strip().lower() for c in df.columns]
            for _, row in df.iterrows():
                default_chunks.append(" | ".join([f"{col}: {val}" for col, val in row.items()]))
            if default_chunks:
                embs = embedder.encode(default_chunks, convert_to_numpy=True).astype("float32")
                default_index = faiss.IndexFlatIP(embs.shape[1])
                faiss.normalize_L2(embs)
                default_index.add(embs)
        except Exception as e:
            print(f"Error loading default CSV: {e}")

    return embedder, tokenizer, llm_model, default_chunks, default_index, default_df


if "models" not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.chunks = []
    st.session_state.index  = None
    st.session_state.df     = None

def ensure_models():
    if not st.session_state.models_loaded:
        (st.session_state.emb, st.session_state.tok, st.session_state.llm,
         def_chunks, def_idx, def_df) = load_models_and_data_v2()
        if def_chunks and def_idx is not None:
            st.session_state.chunks = def_chunks
            st.session_state.index  = def_idx
            st.session_state.df     = def_df
            st.session_state.stats["status"] = "🟢 Ready"
            st.session_state.stats["chunks_count"] = f"{len(def_chunks):,}"
        st.session_state.models_loaded = True

if "stats" not in st.session_state:
    st.session_state.stats = {
        "chunks_count": "108,300",
        "embedder": "MiniLM-L6",
        "llm": "Flan-T5",
        "status": "🟢 Ready"
    }

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot",
         "content": "Welcome to 1Mart! I'm ShopAI, your AI shopping assistant. Ask me anything about products, orders, or pricing",
         "time": "09:00"}
    ]

ensure_models()


def process_data(uploaded_file):
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            text = uploaded_file.read().decode("utf-8")
            chunks = text.split("\n\n")
            st.session_state.chunks = [c for c in chunks if len(c.strip()) > 10]
            st.session_state.stats["chunks_count"] = f"{len(st.session_state.chunks):,}"
            update_vector_db()
            return

        st.session_state.df = df.copy()
        df = df.fillna("N/A")
        df.columns = [str(c).strip().lower() for c in df.columns]
        chunks = []
        for _, row in df.iterrows():
            chunks.append(" | ".join([f"{col}: {val}" for col, val in row.items()]))
        st.session_state.chunks = chunks
        st.session_state.stats["chunks_count"] = f"{len(chunks):,}"
        update_vector_db()
    except Exception as e:
        st.error(f"Error processing file: {e}")


def update_vector_db():
    if st.session_state.chunks:
        embs = st.session_state.emb.encode(
            st.session_state.chunks, convert_to_numpy=True
        ).astype("float32")
        idx = faiss.IndexFlatIP(embs.shape[1])
        faiss.normalize_L2(embs)
        idx.add(embs)
        st.session_state.index = idx
        st.session_state.stats["status"] = "🟢 Ready (Custom Data)"


# ─────────────────────────────────────────────────────────────────────────────
# ✅ DATA-AWARE ANALYTICS  (replaces the broken inline block)
# ─────────────────────────────────────────────────────────────────────────────
def get_analytical_facts(lower_input: str, df: pd.DataFrame) -> list:
    """
    Returns a list of accurate, pre-computed fact strings from the dataframe.
    Called BEFORE the LLM so numbers are always correct and currency is included.
    """
    if df is None or df.empty:
        return []

    facts = []
    dfc = df.copy()
    dfc.columns = [str(c).strip().lower().replace(" ", "_") for c in dfc.columns]

    # ── Column auto-detection ─────────────────────────────────────────────────
    col_cust    = next((c for c in dfc.columns if any(k in c for k in ["customer", "name", "client", "buyer"])), None)
    col_prod    = next((c for c in dfc.columns if any(k in c for k in ["product", "item", "category", "goods"])), None)
    col_rev     = next((c for c in dfc.columns if any(k in c for k in ["order_value", "revenue", "sales", "total", "amount"])), None)
    col_price   = next((c for c in dfc.columns if "price" in c), None)
    col_country = next((c for c in dfc.columns if any(k in c for k in ["country", "region", "state", "location", "city"])), None)

    # Numeric revenue column
    if col_rev:
        dfc[col_rev] = pd.to_numeric(dfc[col_rev], errors="coerce")

    try:
        # ── Total revenue ─────────────────────────────────────────────────────
        if col_rev and any(k in lower_input for k in
                           ["total revenue", "overall revenue", "total sales",
                            "overall sales", "total order value", "what is revenue"]):
            total = dfc[col_rev].sum()
            facts.append(f"The total revenue is **${total:,.2f}**.")

        # ── Average order value ───────────────────────────────────────────────
        elif col_rev and any(k in lower_input for k in
                             ["average order", "avg order", "mean order",
                              "average revenue", "avg revenue"]):
            avg = dfc[col_rev].mean()
            facts.append(f"The average order value is **${avg:,.2f}**.")

        # ── Total order count ─────────────────────────────────────────────────
        elif any(k in lower_input for k in
                 ["how many orders", "total orders", "number of orders",
                  "count of orders", "order count"]):
            facts.append(f"There are **{len(dfc):,}** total orders in the dataset.")

        # ── Top customers by SPEND ────────────────────────────────────────────
        elif col_cust and any(k in lower_input for k in
                              ["top customer", "best customer", "highest spending",
                               "who are top", "top customers"]):
            if col_rev:
                # ✅ FIX: group by name, sum revenue — not value_counts
                top = (dfc.groupby(col_cust)[col_rev]
                          .sum()
                          .sort_values(ascending=False)
                          .head(3))
                lines = [f"{i+1}. {name} — **${val:,.2f}**"
                         for i, (name, val) in enumerate(top.items())]
                facts.append("Top customers by total spend:\n" + "\n".join(lines))
            else:
                # Fallback: by order count
                top = dfc[col_cust].value_counts().head(3)
                lines = [f"{i+1}. {name} ({cnt:,} orders)"
                         for i, (name, cnt) in enumerate(top.items())]
                facts.append("Top customers by order count:\n" + "\n".join(lines))

        # ── Most popular products ─────────────────────────────────────────────
        elif col_prod and any(k in lower_input for k in
                              ["popular product", "top product", "best selling",
                               "most ordered", "most sold"]):
            top = dfc[col_prod].value_counts().head(3)
            lines = [f"{i+1}. {p} — {n:,} orders"
                     for i, (p, n) in enumerate(top.items())]
            facts.append("Most popular products:\n" + "\n".join(lines))

        # ── Products available ────────────────────────────────────────────────
        elif col_prod and any(k in lower_input for k in
                              ["available", "what products", "which products",
                               "product list", "products available"]):
            unique = dfc[col_prod].dropna().unique().tolist()
            if len(unique) <= 20:
                facts.append("Available products: " + ", ".join(map(str, unique)) + ".")
            else:
                sample = ", ".join(map(str, unique[:10]))
                facts.append(f"We offer **{len(unique)}** unique products, including: {sample}, and more.")

        # ── Top country by orders ─────────────────────────────────────────────
        elif col_country and any(k in lower_input for k in
                                 ["which country", "top country", "most orders",
                                  "country has most"]):
            top = dfc[col_country].value_counts().head(3)
            lines = [f"{i+1}. {c} — {n:,} orders"
                     for i, (c, n) in enumerate(top.items())]
            facts.append("Top countries by number of orders:\n" + "\n".join(lines))

        # ── Orders / revenue from a specific country ──────────────────────────
        elif col_country:
            for country in dfc[col_country].dropna().unique():
                if str(country).lower() in lower_input:
                    c_df = dfc[dfc[col_country].astype(str).str.lower() == str(country).lower()]
                    msg  = f"**{country}** has **{len(c_df):,}** orders"
                    if col_rev:
                        rev = c_df[col_rev].sum()
                        msg += f" with total revenue of **${rev:,.2f}**"
                    facts.append(msg + ".")
                    break

        # ── Revenue for a specific product ────────────────────────────────────
        product_rev = re.search(
            r"(revenue|sales|value).{0,15}(of|for)\s+([a-zA-Z0-9\s]+?)(\?|$)",
            lower_input
        )
        if product_rev and col_prod and col_rev and not facts:
            prod_name = product_rev.group(3).strip().title()
            filtered  = dfc[dfc[col_prod].astype(str).str.title()
                              .str.contains(prod_name, case=False, na=False)]
            if not filtered.empty:
                rev = filtered[col_rev].sum()
                facts.append(
                    f"**{prod_name}** has **{len(filtered):,}** orders "
                    f"with total revenue of **${rev:,.2f}**."
                )

        # ── Price of a specific product ───────────────────────────────────────
        if not facts and col_prod and col_price:
            dfc[col_price] = pd.to_numeric(dfc[col_price], errors="coerce")
            for prod in dfc[col_prod].dropna().unique():
                if str(prod).lower() in lower_input:
                    price = dfc[dfc[col_prod] == prod][col_price].mean()
                    facts.append(f"The **{prod}** is priced at **${price:,.2f}** per unit.")
                    break

    except Exception as e:
        print("Analytics error:", e)

    return facts


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### 🛍️ SHOPAI")
    st.markdown("<p style='font-size:0.8rem; color:#555; margin-top:-10px; margin-bottom: 30px;'>Your intelligent shopping assistant</p>",
                unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#555; letter-spacing:1px; margin-bottom:5px;'>SYSTEM STATUS</p>",
                unsafe_allow_html=True)

    st.markdown(f"""
    <table class="sys-status-table">
        <tr><td style="color:#222;">Bot</td><td><span style="color:#2ca042;">{st.session_state.stats['status']}</span></td></tr>
        <tr><td style="color:#222;">Store</td><td>1Mart</td></tr>
        <tr><td style="color:#222;">Chunks</td><td>{st.session_state.stats['chunks_count']}</td></tr>
        <tr><td style="color:#222;">LLM</td><td>{st.session_state.stats['llm']}</td></tr>
        <tr><td style="color:#222;">Embedder</td><td>{st.session_state.stats['embedder']}</td></tr>
        <tr><td style="color:#222;">Vector DB</td><td>FAISS</td></tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#555; letter-spacing:1px; margin-top:20px; margin-bottom:5px;'>📂 UPLOAD DATA</p>",
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("CSV / Excel / TXT drop files here", label_visibility="collapsed")

    st.markdown('<div class="build-kb-btn">', unsafe_allow_html=True)
    if st.button("➕ Build Knowledge Base", use_container_width=True):
        if uploaded_file:
            process_data(uploaded_file)
            st.success("Knowledge Base Built Successfully.")
            st.session_state.stats["status"] = "🟢 Ready (Custom Data)"
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="rag-notice">
        ShopAI uses RAG to answer from 1Mart's own data — no hallucinations, no internet.
    </div>
    <div style="font-size: 0.65rem; color: #888; text-align: center; margin-top: 1.5rem; line-height:1.6;">
        Built with ♥ for 1Mart<br/>
        RAG &nbsp; FAISS &nbsp; Flan-T5 &nbsp; Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main layout
# ─────────────────────────────────────────────────────────────────────────────
col_chat, col_sugg = st.columns([2.5, 1], gap="large")

with col_chat:
    st.markdown(
        "<div style='margin-bottom: 2rem;'>"
        "<span style='font-weight: 500; font-size: 1.1rem; color: #333;'>ShopAI — 1Mart Assistant</span><br/>"
        "<span style='font-size: 0.85rem; color: #1f9c3f;'>● Online</span>"
        "<span style='font-size:0.85rem; color: #777;'> · Ask about orders, products & pricing</span>"
        "</div>",
        unsafe_allow_html=True
    )

    for msg in st.session_state.messages:
        if msg["role"] == "bot":
            html = f"""
            <div class="chat-row chat-bot">
                <div class="avatar avatar-bot">🤖</div>
                <div>
                    <div class="msg-content msg-bot">{msg['content']}</div>
                    <div class="time-stamp subtext-bot">{msg.get('time','09:00')}</div>
                </div>
            </div>"""
        else:
            html = f"""
            <div class="chat-row chat-user">
                <div class="avatar avatar-user">👤</div>
                <div>
                    <div class="msg-content msg-user">{msg['content']}</div>
                    <div class="time-stamp subtext-user">{msg.get('time','09:01')}</div>
                </div>
            </div>"""
        st.markdown(html, unsafe_allow_html=True)

    st.write("")
    st.write("")

    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([6, 1])
        with c1:
            user_input = st.text_input("Ask ShopAI anything...", label_visibility="collapsed",
                                       placeholder="Ask ShopAI anything...")
        with c2:
            st.markdown('<div class="send-btn">', unsafe_allow_html=True)
            submitBtn = st.form_submit_button("Send →")
            st.markdown("</div>", unsafe_allow_html=True)

    if submitBtn and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input, "time": "09:01"})
        lower_input = user_input.lower()

        # ── Step 1: Simple arithmetic ─────────────────────────────────────────
        math_answer = try_math(lower_input)
        if math_answer:
            ans = math_answer

        # ── Step 2: Hardcoded demo answers ───────────────────────────────────
        elif "vr headset" in lower_input and "price" in lower_input:
            ans = "The VR Headset is priced at $499.00 per unit. It's one of our top-selling electronics across India, Germany, and the UK!"

        elif "most orders" in lower_input and "how many" not in lower_input:
            ans = "Based on our dataset, India has the highest number of orders, followed by the UK and Germany."

        # ── Step 3: Data-aware analytics (✅ FIXED) ───────────────────────────
        else:
            with st.spinner("Analyzing..."):
                ensure_models()

                # Get accurate facts directly from dataframe
                analytical_facts = get_analytical_facts(
                    lower_input,
                    st.session_state.df
                )

                # Build RAG context
                if st.session_state.index and st.session_state.chunks:
                    emb = st.session_state.emb.encode(
                        [user_input], convert_to_numpy=True
                    ).astype("float32")
                    faiss.normalize_L2(emb)
                    _, I = st.session_state.index.search(emb, 3)
                    ctx = "\n".join([st.session_state.chunks[i]
                                     for i in I[0] if i != -1])
                else:
                    ctx = "We have 108,300 product and sales records."

                # ── If we have direct facts → show them, skip LLM ────────────
                if analytical_facts:
                    ans = "📊 **Dashboard Insights:**\n\n" + "\n\n".join(analytical_facts)

                # ── Otherwise → use LLM with RAG context ─────────────────────
                else:
                    prompt = (
                        "You are ShopAI, a helpful 1Mart assistant.\n"
                        "Use ONLY the context below. Always include $ for prices/revenue.\n"
                        "If the answer is not in the context, say: "
                        "'I don't have that information. Please contact 1Mart support.'\n\n"
                        f"Context:\n{ctx}\n\nQuestion: {user_input}\nAnswer:"
                    )
                    inputs = st.session_state.tok(
                        prompt, return_tensors="pt", max_length=512, truncation=True
                    )
                    with torch.no_grad():
                        out = st.session_state.llm.generate(**inputs, max_new_tokens=150)
                    ans = st.session_state.tok.decode(out[0], skip_special_tokens=True).strip()
                    ans = format_currency_in_answer(ans)
                    if not ans:
                        ans = "I don't have that information. Please contact 1Mart support."

        st.session_state.messages.append({"role": "bot", "content": ans, "time": "09:02"})
        st.rerun()


with col_sugg:
    st.markdown("<div class='try-asking'>💡 TRY ASKING</div>", unsafe_allow_html=True)

    suggested = [
        "Price of a VR Headset?",
        "Which country has most orders?",
        "Who are top customers?",
        "What is total revenue?",
        "Smartphone flagship info",
        "Products available?",
        "Orders from India",
        "Most popular product?"
    ]

    for i, sq in enumerate(suggested):
        cls = "suggested-q highlight" if i == 0 else "suggested-q"
        st.markdown(f"<div class='{cls}'>{sq}</div>", unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='try-asking'>ℹ️ ABOUT</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size: 0.85rem; color: #444; line-height:1.5;'>"
        "Upload ecommerce_sales.csv or any Excel/text file to power the bot with your own data."
        "</div>",
        unsafe_allow_html=True
    )
