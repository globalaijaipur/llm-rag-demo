
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv


load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# 1) Build the prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# 2) Streamlit UI
st.title("LangChain Demo With LLAMA2 API")
input_text = st.text_input("Enter your question")

# 3) New OllamaLLM instantiation
llm = OllamaLLM(
    model="llama3.2",
    base_url="http://localhost:11434",               # omit if default
    client_kwargs={}                                 # add headers here if you proxy-auth
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# 4) Run it
if input_text:
    answer = chain.invoke({"question": input_text})
    st.write(answer)
