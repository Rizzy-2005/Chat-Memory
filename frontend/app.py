"""
app.py — Streamlit frontend for Chat Memory.
Single-page app; calls the FastAPI backend's HTTP endpoints.
Stage 1: placeholder UI. Later stages will wire in upload, query, and summarize flows.
"""
import streamlit as st

st.set_page_config(page_title="Chat Memory", page_icon="💬")

st.title("💬 Chat Memory")
st.caption("RAG chatbot over your WhatsApp export")

st.info(
    "🚧 Coming soon — upload your WhatsApp export and ask questions about your chats.",
    icon="🚧",
)

# TODO (Stage 6): implement upload widget, query input, and summarize date-range picker.
