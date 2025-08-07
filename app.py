from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please set it in your .env file.")

# Optional LangSmith integration
os.environ["GEMINI_API_KEY"] = api_key
os.environ["LANGSMITH_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# Custom styling (optional)
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# LangChain prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. For each user question, follow this format:\n\n"
               "**Sources:**\n"
               "- [Site Name](https://example.com)\n"
               "- [Site Name](https://example.com)\n\n"
               "**Videos:**\n"
               "- [Video Title](https://youtube.com/...)\n\n"
               "**Answer:**\n"
               "Provide a detailed answer here based on the above sources."),
    ("user", "{question}")
])

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=1.0,
    max_retries=2,
    google_api_key=api_key,
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit UI
st.title("📚 Chatboat: Smart AI")

# Keep Q&A history
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []

# Function to split response
def split_response(response):
    sources = re.search(r"\*\*Sources:\*\*\s*(.*?)\n\s*\*\*", response, re.DOTALL)
    videos = re.search(r"\*\*Videos:\*\*\s*(.*?)\n\s*\*\*", response, re.DOTALL)
    answer = re.search(r"\*\*Answer:\*\*\s*(.*)", response, re.DOTALL)

    sources_text = sources.group(1).strip() if sources else ""
    videos_text = videos.group(1).strip() if videos else ""
    answer_text = answer.group(1).strip() if answer else response.strip()
    return sources_text, videos_text, answer_text

# Loop over existing Q&A + allow next input
for idx in range(len(st.session_state.qa_pairs) + 1):
    if idx < len(st.session_state.qa_pairs):
        q, raw_response = st.session_state.qa_pairs[idx]
        st.markdown(f"### ❓ Question {idx+1}")
        st.markdown(f"{q}")

        sources_text, videos_text, answer_text = split_response(raw_response)

        if sources_text:
            st.markdown("**🔗 Sources:**")
            st.markdown(sources_text)
        if videos_text:
            st.markdown("**🎥 Videos:**")
            st.markdown(videos_text)
        st.markdown("**🧠 Answer:**")
        st.markdown(answer_text)
        st.markdown("---")
    else:
        question = st.text_input(f"Ask question #{idx + 1}", key=f"question_{idx}")
        if question:
            with st.spinner("Thinking..."):
                response = chain.invoke({"question": question})
            st.session_state.qa_pairs.append((question, response))
            st.rerun()
        break
