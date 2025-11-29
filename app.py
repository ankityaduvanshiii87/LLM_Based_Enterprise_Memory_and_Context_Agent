from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import chromadb

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from src.helper import download_embedding
from src.prompt import system_prompt

load_dotenv()

app = Flask(__name__)

# -------------------------
# Load Embeddings
# -------------------------
embeddings = download_embedding()

# -------------------------
# Load ChromaDB
# -------------------------
chroma_client = chromadb.PersistentClient(path="chromadb_store")

vector_db = Chroma(
    client=chroma_client,
    collection_name="medical_rag",
    embedding_function=embeddings
)

retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# -------------------------
# Load LLaMA 3.2 1B Instruct Model
# -------------------------
model_name = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

llm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto"
)

llm_pipeline = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.6,
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

# -------------------------
# RAG Prompt Template
# -------------------------
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=f"""
{system_prompt}

Context:
{{context}}

Question: {{question}}

Answer:
"""
)

# -------------------------
# RAG Chain
# -------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)


# -------------------------
# SAVE HISTORY FUNCTION
# -------------------------
def save_history(query, response):
    with open("history.txt", "a", encoding="utf-8") as f:
        f.write("User: " + query + "\n")
        f.write("Assistant: " + response + "\n")
        f.write("-" * 50 + "\n")


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]

    result = qa_chain.invoke({"query": user_msg})
    answer = result["result"]

    # SAVE Q/A HISTORY
    save_history(user_msg, answer)

    return answer

@app.route("/history")
def history():
    if not os.path.exists("history.txt"):
        return render_template("history.html", history=[])

    with open("history.txt", "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    return render_template("history.html", history=lines)

if __name__ == "__main__":
    app.run(port=8080, debug=True)
