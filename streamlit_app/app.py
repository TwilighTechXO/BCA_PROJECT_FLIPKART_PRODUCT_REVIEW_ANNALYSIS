import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from io import StringIO

st.set_page_config(page_title="Customer Feedback Dashboard", layout="wide")

# ----------------------------
# PREMIUM CSS
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
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("final_model.pkl", "rb"))
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# ----------------------------
# EMBEDDED SAMPLE DATA
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
    if file:
        df = pd.read_csv(file)
    else:
        df = None

# ----------------------------
# PROCESS
# ----------------------------
if 'df' in locals() and df is not None:

    if "review" not in df.columns:
        st.error("CSV must contain 'review' column")
    else:
        X = vectorizer.transform(df["review"].astype(str))
        df["sentiment"] = model.predict(X)

        # ----------------------------
        # PRODUCT FILTER
        # ----------------------------
        product = st.selectbox("Select Product", ["All"] + list(df["product_name"].unique()))
        if product != "All":
            df = df[df["product_name"] == product]

        total = len(df)
        pos = (df["sentiment"] == "positive").sum()
        neg = (df["sentiment"] == "negative").sum()
        neu = (df["sentiment"] == "neutral").sum()

        # ----------------------------
        # KPI CARDS
        # ----------------------------
        col1, col2, col3, col4 = st.columns(4)

        col1.markdown(f'<div class="kpi kpi-total">Total<br>{total}</div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="kpi kpi-pos">Positive<br>{round(pos/total*100,1)}%</div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="kpi kpi-neg">Negative<br>{round(neg/total*100,1)}%</div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="kpi kpi-neu">Neutral<br>{round(neu/total*100,1)}%</div>', unsafe_allow_html=True)

        # ----------------------------
        # TABS
        # ----------------------------
        tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Insights", "Text Analysis", "Business Suggestions"])

        # ============================
        # OVERVIEW
        # ============================
        with tab1:

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Sentiment Distribution")
                fig, ax = plt.subplots()
                df["sentiment"].value_counts().plot(
                    kind="bar",
                    ax=ax,
                    color=["#10b981", "#ef4444", "#9ca3af"]
                )
                st.pyplot(fig)

            with col2:
                st.subheader("Sentiment Share")
                fig, ax = plt.subplots()
                df["sentiment"].value_counts().plot(
                    kind="pie",
                    autopct="%1.1f%%",
                    colors=["#10b981", "#ef4444", "#9ca3af"],
                    ax=ax
                )
                ax.set_ylabel("")
                st.pyplot(fig)

        # ============================
        # INSIGHTS
        # ============================
        with tab2:

            if "rating" in df.columns:
                st.subheader("Sentiment vs Rating")
                st.bar_chart(pd.crosstab(df["rating"], df["sentiment"]))

            df["review_length"] = df["review"].apply(lambda x: len(str(x).split()))
            st.subheader("Review Length vs Sentiment")
            st.bar_chart(df.groupby("sentiment")["review_length"].mean())

            st.subheader("Low Rating Alerts")
            st.dataframe(df[(df["rating"] <= 2) & (df["sentiment"] == "negative")])

        # ============================
        # TEXT ANALYSIS
        # ============================
        with tab3:

            text = " ".join(df["review"].astype(str))

            st.subheader("Word Cloud")
            wc = WordCloud(width=900, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

            words = Counter(text.split()).most_common(10)
            st.subheader("Top Keywords")
            st.bar_chart(pd.DataFrame(words, columns=["Word","Count"]).set_index("Word"))

            pos_words = Counter(" ".join(df[df["sentiment"]=="positive"]["review"]).split()).most_common(10)
            st.subheader("Top Positive Words")
            st.bar_chart(pd.DataFrame(pos_words, columns=["Word","Count"]).set_index("Word"))

            neg_words = Counter(" ".join(df[df["sentiment"]=="negative"]["review"]).split()).most_common(10)
            st.subheader("Top Negative Words")
            st.bar_chart(pd.DataFrame(neg_words, columns=["Word","Count"]).set_index("Word"))

        # ============================
        # BUSINESS SUGGESTIONS
        # ============================
        with tab4:

            st.subheader("Automated Business Insights")

            if neg > pos:
                st.write("• High negative sentiment detected. Improve product quality.")
            if "bad" in text:
                st.write("• Customers frequently mention quality issues.")
            if pos > neg:
                st.write("• Overall customer satisfaction is strong.")
            if total > 0:
                st.write("• Continue monitoring customer feedback trends.")

        # ----------------------------
        # DATA PREVIEW
        # ----------------------------
        st.subheader("Data Preview")
        st.dataframe(df.head())
