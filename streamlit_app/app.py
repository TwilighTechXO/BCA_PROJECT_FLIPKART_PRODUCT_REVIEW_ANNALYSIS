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

st.set_page_config(page_title="Customer Feedback Dashboard", layout="wide")

# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>
body { background-color: #f4f6f9; }
h1 { color: #1f2937; font-weight: 700; }
.kpi {
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-weight: bold;
}
.kpi-total { background: linear-gradient(135deg, #3b82f6, #6366f1); }
.kpi-pos { background: linear-gradient(135deg, #10b981, #34d399); }
.kpi-neg { background: linear-gradient(135deg, #ef4444, #f87171); }
.kpi-neu { background: linear-gradient(135deg, #6b7280, #9ca3af); }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD MODEL (FROM GDRIVE)
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

# ----------------------------
# HEADER
# ----------------------------
st.title("Customer Feedback Analysis Dashboard")
st.write("AI-powered insights for product reviews")

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("Configuration")
option = st.sidebar.radio("Data Source", ["Use Sample Data", "Upload CSV"])

if option == "Use Sample Data":
    df = pd.read_csv(StringIO(sample_data))
elif option == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    df = pd.read_csv(file) if file else None

# ----------------------------
# MAIN PROCESS
# ----------------------------
if 'df' in locals() and df is not None:

    df.columns = df.columns.str.strip().str.lower()

    # ----------------------------
    # AUTO COLUMN DETECTION
    # ----------------------------
    review_col = None
    for col in df.columns:
        if col in ["review", "reviews", "text", "comment", "feedback"]:
            review_col = col
            break

    if review_col is None:
        st.error("No review column found")
        st.stop()

    rating_col = None
    for col in df.columns:
        if "rating" in col:
            rating_col = col
            break

    product_col = None
    for col in df.columns:
        if "product" in col or "name" in col:
            product_col = col
            break

    # ----------------------------
    # CLEAN PRODUCT NAME (FIX ????)
    # ----------------------------
    def clean_product_name(name):
        name = str(name)
        name = name.split("(")[0]  # remove extra description
        name = re.sub(r'[^a-zA-Z0-9\s]', '', name)  # remove weird chars
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    if product_col:
        df["product_short"] = df[product_col].apply(clean_product_name)
    else:
        df["product_short"] = "Unknown"

    # ----------------------------
    # CLEAN DATA
    # ----------------------------
    df[review_col] = df[review_col].astype(str)

    if rating_col:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")

    # ----------------------------
    # MODEL PREDICTION
    # ----------------------------
    X = vectorizer.transform(df[review_col])
    df["sentiment"] = model.predict(X)

    # ----------------------------
    # FILTER
    # ----------------------------
    product = st.selectbox("Select Product", ["All"] + list(df["product_short"].dropna().unique()))
    if product != "All":
        df = df[df["product_short"] == product]

    # ----------------------------
    # KPI
    # ----------------------------
    total = len(df)
    pos = (df["sentiment"] == "positive").sum()
    neg = (df["sentiment"] == "negative").sum()
    neu = (df["sentiment"] == "neutral").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f'<div class="kpi kpi-total">Total<br>{total}</div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="kpi kpi-pos">Positive<br>{round((pos/total)*100,1) if total else 0}%</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="kpi kpi-neg">Negative<br>{round((neg/total)*100,1) if total else 0}%</div>', unsafe_allow_html=True)
    col4.markdown(f'<div class="kpi kpi-neu">Neutral<br>{round((neu/total)*100,1) if total else 0}%</div>', unsafe_allow_html=True)

    # ----------------------------
    # TABS
    # ----------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Insights", "Text Analysis", "Business Suggestions"])

    # OVERVIEW
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            df["sentiment"].value_counts().plot(
                kind="bar", ax=ax,
                color=["#10b981", "#ef4444", "#9ca3af"]
            )
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            df["sentiment"].value_counts().plot(
                kind="pie", autopct="%1.1f%%",
                colors=["#10b981", "#ef4444", "#9ca3af"],
                ax=ax
            )
            ax.set_ylabel("")
            st.pyplot(fig)

    # INSIGHTS
    with tab2:
        if rating_col:
            st.subheader("Sentiment vs Rating")
            st.bar_chart(pd.crosstab(df[rating_col], df["sentiment"]))

            st.subheader("Low Rating Alerts")
            low_df = df[(df[rating_col].notna()) & (df[rating_col] <= 2) & (df["sentiment"] == "negative")]
            st.dataframe(low_df)

        df["review_length"] = df[review_col].apply(lambda x: len(str(x).split()))
        st.subheader("Review Length vs Sentiment")
        st.bar_chart(df.groupby("sentiment")["review_length"].mean())

    # TEXT ANALYSIS
    with tab3:
        text = " ".join(df[review_col])

        wc = WordCloud(width=900, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

        words = Counter(text.split()).most_common(10)
        st.bar_chart(pd.DataFrame(words, columns=["Word", "Count"]).set_index("Word"))

    # BUSINESS INSIGHTS
    with tab4:
        st.subheader("Automated Business Insights")

        if total > 0:
            if (neg/total)*100 > 30:
                st.write("• High negative sentiment detected. Improve product quality.")
            if (pos/total)*100 > 70:
                st.write("• Customers are highly satisfied.")
            if "bad" in text:
                st.write("• Quality issues frequently reported.")
            if "good" in text:
                st.write("• Strong positive feedback on performance.")

    # DATA
    st.subheader("Data Preview")
    st.dataframe(df.head())
