import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("AIzaSyBBDr9CopHtJIiPTWhTZP-37MWkTwi0rVM"))

st.set_page_config(page_title="🩺 Medical Q&A Chatbot", page_icon="💊")
st.title("💊 Medical Q&A Chatbot")
st.write("Ask health-related questions — powered by Gemini + MedQuAD")

df = pd.read_csv(
    "C:\\Users\\ASUS\\Downloads\\null class\\task3_medical_bot\\medquad (3).csv")
df = df.dropna(subset=["question", "answer"])

question = st.text_input("Type your medical question:")
if st.button("Get Answer"):
    vectorizer = TfidfVectorizer().fit(df["question"])
    user_vec = vectorizer.transform([question])
    data_vecs = vectorizer.transform(df["question"])
    sim = cosine_similarity(user_vec, data_vecs).flatten()
    idx = sim.argmax()

    base_answer = df.iloc[idx]["answer"]

    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest")
    response = model.generate_content(
        f"Explain this medical answer simply: {base_answer}")
    st.subheader("🧠 AI Response")
    st.write(response.text)
