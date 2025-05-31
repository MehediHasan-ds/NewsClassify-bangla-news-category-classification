import streamlit as st
import requests

st.set_page_config(page_title="Bangla News Classifier", layout="centered")
st.title("📰 Bangla News Category Classifier")

st.write("Enter a Bangla news headline/paragraph or upload a `.txt` file for prediction.")

# Text input
text_input = st.text_area("✍️ Paste your Bangla news text below:", height=150)

# File upload
uploaded_file = st.file_uploader("📄 Or upload a .txt file", type=["txt"])

input_text = ""

# Use uploaded file if available
if uploaded_file is not None:
    file_content = uploaded_file.read().decode("utf-8")
    input_text = file_content.strip()

# If user typed text, it overrides file content
if text_input.strip():
    input_text = text_input.strip()

if st.button("Predict"):
    if input_text == "":
        st.warning("⚠️ Please provide text or upload a file.")
    else:
        with st.spinner("🔍 Predicting..."):
            try:
                response = requests.post("http://localhost:8000/predict", json={"text": input_text})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"🧠 Predicted Category: **{result['category'].capitalize()}**")
                else:
                    st.error("🚫 API error: Something went wrong.")
            except Exception as e:
                st.error(f"🔌 Could not connect to backend: {e}")
