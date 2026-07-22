import streamlit as st
import pickle
import pandas as pd
import re
import string
from PyPDF2 import PdfReader

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Resume Screening",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_models():
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    return model, tfidf, le


model, tfidf, le = load_models()

# -----------------------------
# Resume Cleaning Function
# (Replace with your notebook's
# function if different)
# -----------------------------
def cleanResume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"www\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"[^A-Za-z ]", " ", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# PDF Reader
# -----------------------------
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📄 Resume Screening")

st.sidebar.markdown(
"""
### About

This application predicts the most suitable **Job Category** from a resume using a Machine Learning model.

### Model

- Random Forest
- TF-IDF Vectorizer
- Scikit-Learn

### Supported Files

- PDF
- TXT

"""
)

st.sidebar.info(
    "Paste your resume or upload a file and click **Predict Category**."
)

# -----------------------------
# Main Title
# -----------------------------
st.title("📄 Resume Screening using Machine Learning")

st.write(
"""
Predict the most suitable **Job Category**
from your resume.
"""
)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2 = st.tabs(
    ["✍ Paste Resume", "📂 Upload Resume"]
)

resume_text = ""

# -----------------------------
# Paste Resume
# -----------------------------
with tab1:

    resume_text = st.text_area(
        "Paste Resume Here",
        height=350,
        placeholder="Paste your complete resume..."
    )

# -----------------------------
# Upload Resume
# -----------------------------
with tab2:

    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "txt"]
    )

    if uploaded_file is not None:

        if uploaded_file.type == "application/pdf":
            resume_text = extract_pdf_text(uploaded_file)

        elif uploaded_file.type == "text/plain":
            resume_text = uploaded_file.read().decode("utf-8")

        st.success("✅ Resume Uploaded Successfully!")

        with st.expander("Preview Resume"):

            st.write(resume_text[:3000])

# -----------------------------
# Buttons
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    predict = st.button(
        "🔍 Predict Category",
        use_container_width=True
    )

with col2:
    clear = st.button(
        "🧹 Clear",
        use_container_width=True
    )

if clear:
    st.rerun()

# -----------------------------
# Prediction
# -----------------------------
if predict:

    if resume_text.strip() == "":

        st.warning("Please paste or upload a resume.")

    else:

        cleaned_resume = cleanResume(resume_text)

        vector = tfidf.transform([cleaned_resume])

        prediction = model.predict(vector)

        category = le.inverse_transform(prediction)[0]

        st.success(f"🎯 Predicted Category: **{category}**")

        # -----------------------------
        # Confidence
        # -----------------------------
        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(vector)[0]

            confidence = probabilities.max() * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

            df = pd.DataFrame({
                "Category": le.classes_,
                "Probability": probabilities
            })

            df = df.sort_values(
                by="Probability",
                ascending=False
            )

            st.subheader("Prediction Probabilities")

            st.bar_chart(
                df.set_index("Category")
            )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
    "Developed using Streamlit • Random Forest • TF-IDF • Scikit-Learn"
)