from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

os.environ["GEMINI_API_KEY"] = api_key
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# LangChain setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user's queries."),
    ("user", "{question}")
])
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

st.title("Chatboat")

if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []

for idx in range(len(st.session_state.qa_pairs) + 1):
    if idx < len(st.session_state.qa_pairs):
        q, a = st.session_state.qa_pairs[idx]
        st.markdown(f"**Q{idx+1}:** {q}")
        st.markdown(f"**A{idx+1}:** {a}")
        st.markdown("---")
    else:
        question = st.text_input(f"Ask question #{idx + 1}", key=f"question_{idx}")
        if question:
            response = chain.invoke({"question": question})
            st.session_state.qa_pairs.append((question, response))
            st.rerun()
        break
