import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000"

st.set_page_config(page_title="GenAI RAG App", layout="centered")

st.title("🧠 Generative AI RAG App")
st.write("Ask questions about your uploaded documents.")

# ---- Document Upload ----
st.subheader("📄 Upload a Document")

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    files = {"file": uploaded_file}
    response = requests.post(f"{API_URL}/upload", files=files)

    if response.status_code == 200:
        st.success("Document uploaded successfully!")
    else:
        st.error(response.text)

# ---- Ask Question ----
st.subheader("❓ Ask a Question")

question = st.text_input("Enter your question")

if st.button("Ask"):
    if not question:
        st.warning("Please enter a question.")
    else:
        payload = {"question": question}
        response = requests.post(f"{API_URL}/ask", json=payload)

        if response.status_code == 200:
            answer = response.json()["answer"]
            st.markdown("### 🧠 Answer")
            st.write(answer)
        else:
            st.error(response.text)