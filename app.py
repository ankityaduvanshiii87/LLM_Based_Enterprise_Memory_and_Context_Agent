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
# Load LLaMA 2 Model
# -------------------------
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name="microsoft/Phi-3-mini-4k-instruct"
model_name="meta-llama/Llama-3.2-1B-Instruct"
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


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]

    result = qa_chain.invoke({"query": user_msg})
    answer = result["result"]

    return answer


if __name__ == "__main__":
    app.run(port=8080, debug=True)
