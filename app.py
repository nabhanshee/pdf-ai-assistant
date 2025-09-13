from flask import Flask, request
import os
from dotenv import load_dotenv
load_dotenv()

from flask import jsonify


from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from flask import Flask, request, render_template


load_dotenv()
cached_llm = ChatGroq(model_name="llama3-70b-8192")


# Make folders if they don't exist
os.makedirs("pdf", exist_ok=True)
os.makedirs("db", exist_ok=True)

app = Flask(__name__)


embedding = FastEmbedEmbeddings()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template("""
<s>[INST] You are a helpful assistant. Use the information to answer. If not found, say 'Not available'. [/INST] </s>
[INST] {input}
       Context: {context}
       Answer:
[/INST]
""")


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    response = cached_llm.invoke(query)
    print(response)

    return {"answer": response}


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory="db", embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 20, "score_threshold": 0.1}
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})
    print(result)

    sources = []
    for doc in result.get("context", []):
        sources.append({
            "source": doc.metadata.get("source", "unknown"),
            "page_content": doc.page_content
        })

    return jsonify({"answer": result["answer"], "sources": sources})


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="db"
    )

    vector_store.persist()

    return {
        "status": "Uploaded!",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }

@app.route("/")
def index():
    return render_template("index.html")


# @app.route("/")
# def home():
#     return "PDF AI backend is running!"



def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
