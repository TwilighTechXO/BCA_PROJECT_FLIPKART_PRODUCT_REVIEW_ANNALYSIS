## Customer Feedback Analysis Dashboard (Sentify Analysis)

This project is an AI-powered web application built using **Streamlit** that analyzes customer reviews and transforms raw textual feedback into meaningful business insights. The main goal is to help businesses understand customer sentiment, identify issues, and make data-driven decisions in real time.

---

## Model & Approach

The core of this system is a machine learning pipeline based on:

* **Natural Language Processing** (NLP) for understanding textual reviews
* **TF-IDF Vectorization** to convert text into numerical features
* A trained classification model (sentiment analysis) that predicts whether a review is:

  * **Positive**
  * **Negative**
  * **Neutral**

### Model Files (Loaded from Cloud)

To ensure easy deployment and no local dependency issues, the trained model and vectorizer are hosted online and loaded dynamically:

* **Final Model (.pkl):**
  [https://drive.google.com/file/d/10bGGKcgByyeDaI3_g_bCWRroyv5DX9UJ](https://drive.google.com/file/d/10bGGKcgByyeDaI3_g_bCWRroyv5DX9UJ)

* **TF-IDF Vectorizer (.pkl):**
  [https://drive.google.com/file/d/1V6vBxI0nSoJt2sn-4FA2ML-ueOIwFteO](https://drive.google.com/file/d/1V6vBxI0nSoJt2sn-4FA2ML-ueOIwFteO)

* **Dataset:**
  [The dataset used to train the model is collected from the kaggle website](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset?resource=download)

This approach allows the app to run directly on platforms like Streamlit Cloud without requiring local model files.

---

## How It Works

1. The user uploads a dataset (CSV) containing product reviews
2. The system automatically:

   * Detects relevant columns (review, rating, product, etc.)
   * Cleans and preprocesses the data
3. Reviews are transformed using TF-IDF
4. The trained model predicts sentiment for each review
5. Results are displayed through interactive dashboards and graphs

---

## Dashboard Link
* **Sample Dataset:**
  [The sample dataset to test the dashboard](https://drive.google.com/file/d/153y7xAdyY6zTx_JS-ZG05bidZul2MOPV/view?usp=sharing)

* **Streamlit Dashboard:**
  [Dashboard streamlit link](https://bcaprojectfeedbackannalysis-crow9j4bo2luvnduzfyjjn.streamlit.app/)

* **QR CODE for Dashboard:**
  <img width="400" height="600" alt="Dashboard qr code" src="https://github.com/user-attachments/assets/b7f97fc2-9461-4682-aebd-7e2cd3160e73" />

## Dashboard Navigation & Features

The application is designed with a **step-by-step, user-friendly interface** so even non-technical users can easily interact with it.
---

### Step 1: Data Input

Users can:

* Use built-in sample data
* OR upload their own CSV file

The app intelligently adapts to different column names and formats.

---

### Step 2: Product Selection

If multiple products exist, the user selects one product from a dropdown.
This ensures **focused and accurate analysis** instead of mixing all products together.

---

### Step 3: KPI Overview

Key performance indicators are displayed:

* Total reviews
* Positive sentiment %
* Negative sentiment %
* Neutral sentiment %

This gives a quick understanding of customer perception.

---

### Overview Tab

Provides a high-level summary:

* Bar chart of sentiment distribution
* Pie chart showing sentiment share
* Additional distribution insights

---

### Insights Tab

This is the **analytical core** of the application:

* Sentiment vs Rating analysis
* Review length vs sentiment
* Price vs sentiment (auto-detected)
* Rating distribution
* Sentiment trends

The system automatically handles:

* Numeric or text-based price values
* Different column naming formats

---

### Text Analysis Tab

Focuses on understanding the actual feedback content:

* Word Cloud visualization
* Top frequently used keywords
* Helps identify common customer opinions

---

### Next Steps (Business Recommendations)

Instead of static outputs, the app provides **dynamic AI-driven suggestions**, such as:

* Improve product quality (if negative sentiment is high)
* Launch marketing campaigns (if positive sentiment is high)
* Investigate pricing or quality issues based on keywords

This makes the tool **actionable**, not just analytical.

---

## Dashboard Screenshots
<img width="931" height="549" alt="image" src="https://github.com/user-attachments/assets/072fffa4-5964-4cef-aba9-96b6b8d113a2" />
<img width="931" height="546" alt="image" src="https://github.com/user-attachments/assets/4c272603-dadf-4b97-8996-c51dcd0e7fc5" />
<img width="931" height="559" alt="image" src="https://github.com/user-attachments/assets/e2280f60-ac77-496a-abfb-51f70ce91e5a" />






## Key Highlights

✔ Fully dynamic – works with any dataset format
✔ No manual column setup required
✔ Cloud-based model loading
✔ Real-time sentiment prediction
✔ Business-focused insights
✔ Beginner-friendly interface

---

## Conclusion

This project combines **machine learning, data visualization, and user-centric design** to create a powerful tool for customer feedback analysis. It bridges the gap between raw data and actionable insights, making it highly useful for businesses, analysts, and decision-makers.
