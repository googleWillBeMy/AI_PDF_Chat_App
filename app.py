import streamlit as st
import PyPDF2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Initialize the knowledge base and models
knowledge_base = {}
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
question_answering = pipeline("question-answering", model="deepset/roberta-base-squad2")


# Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text()
    return text


# Generate embeddings for text
def generate_embeddings(text):
    return embedding_model.encode([text])[0]


# Find the best-matching document based on question similarity
def find_best_document(question, knowledge_base):
    question_embedding = generate_embeddings(question)
    similarities = {}
    for doc_id, doc_data in knowledge_base.items():
        doc_embedding = doc_data["embedding"]
        similarities[doc_id] = cosine_similarity([question_embedding], [doc_embedding])[0][0]
    best_doc_id = max(similarities, key=similarities.get)
    return knowledge_base[best_doc_id]["text"]


# Use the language model to answer the question based on retrieved context
def get_answer(question, context):
    result = question_answering(question=question, context=context)
    return result['answer']

# Main function to answer questions by combining retrieval and generation
def answer_question(question):
    context = find_best_document(question, knowledge_base)
    return get_answer(question, context)


# Streamlit UI
st.title("AI PDF Chat Application")

# PDF Upload and Knowledge Base Update
uploaded_file = st.file_uploader("Upload PDF", type="pdf")
if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)
    knowledge_base["uploaded_pdf"] = {
        "text": text,
        "embedding": generate_embeddings(text)
    }
    st.success("PDF uploaded and processed successfully!")

# Ask a question
question = st.text_input("Ask a question:")
if st.button("Get Answer") and question:
    answer = answer_question(question)
    st.write("Answer:", answer)

