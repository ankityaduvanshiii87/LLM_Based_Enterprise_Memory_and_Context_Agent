from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

# Local Embeddings & Chroma DB
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# # LLaMA Model
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# # LangChain RAG
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load existing vector DB
vectordb = Chroma(
    persist_directory="db",
    embedding_function=embeddings
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

model_name = "meta-llama/Llama-2-7b-chat-hf"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_8bit=True,     
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2
)


# ---------------------------------------------------------
# LangChain Prompt & RAG Chain
# ---------------------------------------------------------

system_prompt = """You are a helpful medical assistant. You answer using only the retrieved context."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# LLM Wrapper
def llama_invoke(prompt_text):
    result = pipe(prompt_text)[0]['generated_text']
    return result

question_answer_chain = create_stuff_documents_chain(llama_invoke, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# # ---------------------------------------------------------
# # Flask Routes
# # ---------------------------------------------------------

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    user_msg = request.form["msg"]
    print("User:", user_msg)

    response = rag_chain.invoke({"input": user_msg})

    answer = response["answer"]
    print("Bot:", answer)
    return answer


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
