import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from io import StringIO
import os
import urllib.request
import re
from streamlit_lottie import st_lottie
import requests

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Sentify | Customer Intelligence",
    page_icon="✨",
    layout="wide"
)

# ----------------------------
# UI (FONT SIZE INCREASED)
# ----------------------------
st.markdown("""
<style>
html, body {
    font-family: 'Poppins', sans-serif;
}

.stTabs [data-baseweb="tab"] {
    font-size: 18px !important;
    font-weight: 700;
}

.metric-card {
    background: rgba(255, 255, 255, 0.9);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
}

.metric-label { font-size: 16px; }
.metric-value { font-size: 32px; font-weight: 800; }

.total-accent { border-bottom: 5px solid #6366F1; }
.pos-accent { border-bottom: 5px solid #10B981; }
.neg-accent { border-bottom: 5px solid #EF4444; }
.neu-accent { border-bottom: 5px solid #94A3B8; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model_url = "https://drive.google.com/uc?id=10bGGKcgByyeDaI3_g_bCWRroyv5DX9UJ"
    vectorizer_url = "https://drive.google.com/uc?id=1V6vBxI0nSoJt2sn-4FA2ML-ueOIwFteO"

    if not os.path.exists("final_model.pkl"):
        urllib.request.urlretrieve(model_url, "final_model.pkl")

    if not os.path.exists("tfidf_vectorizer.pkl"):
        urllib.request.urlretrieve(vectorizer_url, "tfidf_vectorizer.pkl")

    model = pickle.load(open("final_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.title("Sentify AI")
option = st.sidebar.radio("Data Source", ["Use Sample Data", "Upload CSV"])

# ----------------------------
# SAMPLE DATA
# ----------------------------
sample_data = """product_name,product_price,rating,review
Candes 12L Cooler,3999,5,super product great cooling
Candes 12L Cooler,3999,1,very bad product useless
Candes 12L Cooler,3999,3,average performance ok
Candes 60L Cooler,8999,5,excellent airflow amazing
Candes 60L Cooler,8999,2,bad quality not satisfied
Candes 60L Cooler,8999,5,very nice product happy
Candes 60L Cooler,8999,4,good but can improve
"""

if option == "Use Sample Data":
    df = pd.read_csv(StringIO(sample_data))
else:
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    df = pd.read_csv(file) if file else None

# ----------------------------
# MAIN
# ----------------------------
if df is not None:

    df.columns = df.columns.str.strip().str.lower()

    review_col = next((c for c in df.columns if c in ["review","reviews","text","comment","feedback"]), None)
    rating_col = next((c for c in df.columns if "rating" in c), None)
    product_col = next((c for c in df.columns if "product" in c or "name" in c), None)

    # 🔥 NEW: PRICE COLUMN AUTO DETECT
    price_col = next((c for c in df.columns if "price" in c), None)

    if review_col is None:
        st.error("No review column found")
        st.stop()

    def clean_name(x):
        x = str(x).split("(")[0]
        x = re.sub(r'[^a-zA-Z0-9\s]', '', x)
        return re.sub(r'\s+', ' ', x).strip()

    df["product_short"] = df[product_col].apply(clean_name) if product_col else "Unknown"

    df[review_col] = df[review_col].astype(str)

    if rating_col:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

    if price_col:
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

    X = vectorizer.transform(df[review_col])
    df["sentiment"] = model.predict(X)

    # ----------------------------
    # SELECT PRODUCT
    # ----------------------------
    st.title("Select Product for Analysis")

    product_list = list(df["product_short"].dropna().unique())
    selected_product = st.selectbox("Choose a product", product_list)

    df = df[df["product_short"] == selected_product]

    st.success(f"Showing analysis for: {selected_product}")

    # ----------------------------
    # KPIs
    # ----------------------------
    total = len(df)
    pos = (df["sentiment"]=="positive").sum()
    neg = (df["sentiment"]=="negative").sum()
    neu = (df["sentiment"]=="neutral").sum()

    cols = st.columns(4)

    def card(col, label, value, cls):
        col.markdown(f'<div class="metric-card {cls}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

    card(cols[0], "Total Reviews", total, "total-accent")
    card(cols[1], "Positive %", f"{round((pos/total)*100,1) if total else 0}%", "pos-accent")
    card(cols[2], "Negative %", f"{round((neg/total)*100,1) if total else 0}%", "neg-accent")
    card(cols[3], "Neutral %", f"{round((neu/total)*100,1) if total else 0}%", "neu-accent")

    st.write("---")

    # ----------------------------
    # TABS
    # ----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Insights", "Text Analysis", "Next Steps"])

    # ----------------------------
    # OVERVIEW (UNCHANGED)
    # ----------------------------
    with tab1:
        c1, c2 = st.columns(2)

        with c1:
            fig, ax = plt.subplots()
            df["sentiment"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with c2:
            fig, ax = plt.subplots()
            df["sentiment"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

    # ----------------------------
    # 🔥 INSIGHTS (UPDATED ONLY)
    # ----------------------------
    with tab2:

        # 1️⃣ Sentiment vs Rating
        if rating_col:
            st.subheader("Sentiment vs Rating")
            st.bar_chart(pd.crosstab(df[rating_col], df["sentiment"]))

        # 2️⃣ Review Length
        df["review_length"] = df[review_col].apply(lambda x: len(x.split()))
        st.subheader("Review Length vs Sentiment")
        st.bar_chart(df.groupby("sentiment")["review_length"].mean())

        # 3️⃣ 🔥 NEW: Price Analysis
        if price_col:
            st.subheader("Price vs Sentiment")

            st.bar_chart(df.groupby("sentiment")[price_col].mean())

            st.subheader("Price Distribution")
            st.line_chart(df[price_col].dropna())

        # 4️⃣ 🔥 NEW: Sentiment Count Trend
        st.subheader("Sentiment Count")
        st.line_chart(df["sentiment"].value_counts())

        # 5️⃣ 🔥 NEW: Rating Distribution (if exists)
        if rating_col:
            st.subheader("Rating Distribution")
            st.bar_chart(df[rating_col].value_counts())

    # ----------------------------
    # TEXT ANALYSIS (UNCHANGED)
    # ----------------------------
    with tab3:
        text = " ".join(df[review_col])

        wc = WordCloud(width=800, height=400).generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        words = Counter(text.split()).most_common(10)
        st.bar_chart(pd.DataFrame(words, columns=["Word","Count"]).set_index("Word"))

    # ----------------------------
    # NEXT STEPS (UNCHANGED)
    # ----------------------------
    with tab4:
        st.subheader("AI Business Recommendations")

        if total > 0:
            neg_percent = (neg/total)*100
            pos_percent = (pos/total)*100

            if neg_percent > 40:
                st.error("Improve product quality & support immediately.")
            elif neg_percent > 20:
                st.warning("Investigate customer complaints.")
            else:
                st.success("Maintain current strategy.")

            if pos_percent > 70:
                st.success("Launch marketing campaigns.")
            elif pos_percent > 50:
                st.info("Focus on customer loyalty.")

    with st.expander("View Data"):
        st.dataframe(df.head())

else:
    st.info("Upload a dataset to start analysis")
