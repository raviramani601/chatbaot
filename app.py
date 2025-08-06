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

if api_key is None:
    raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

os.environ["GEMINI_API_KEY"] = api_key
os.environ["LANGSMITH_TRACING_V2"]="true"
os.environ["LANGSMITH_API_KEY"]= os.getenv("LANGSMITH_API_KEY")

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)
st.title('Langchain Demo With GEMINIAI API')
input_text=st.text_input("Ask question")

llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=1.0,
        max_retries=2,
        google_api_key=api_key,
)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))