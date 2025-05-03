

import os
from langchain_ollama import OllamaLLM
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

PDF_PATH = "speech.pdf"  # adjust if needed

def build_qa_chain(pdf_path: str) -> RetrievalQA:
    # 1) Load PDF pages
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"[+] Loaded {len(pages)} page(s) from '{pdf_path}'")

    # 2) Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    print(f"[+] Split into {len(chunks)} chunks")

    # 3) Embed & index with HuggingFace
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("[+] Built FAISS index with HF embeddings")

    # 4) Instantiate Ollama LLM
    llm = OllamaLLM(
        model="llama3.2",
        # base_url="http://localhost:11434",  # only if you changed the default
        client_kwargs={}                       # put proxy-auth headers here if needed
    )
    print("[+] Initialized OllamaLLM")

    # 5) Build retrieval-augmented QA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",            # “stuff” strategy
        retriever=vectorstore.as_retriever(k=4)  # top-4 chunks per query
    )
    print("[+] RetrievalQA chain ready\n")
    return qa

def main():
    if not os.path.isfile(PDF_PATH):
        print(f"❌ PDF not found at '{PDF_PATH}'. Exiting.")
        return

    qa_chain = build_qa_chain(PDF_PATH)
    print("Enter your questions below (type 'exit' or Ctrl-C to quit):\n")

    try:
        while True:
            query = input("Q: ").strip()
            if not query or query.lower() in {"exit", "quit"}:
                break
            answer = qa_chain.run(query)
            print("A:", answer, "\n")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
