import os

os.environ["TRANSFORMERS_NO_TF"] = "1"   # << VERY IMPORTANT

from dotenv import load_dotenv
import os
from src.helper import (
    load_pdf_files,
    filter_to_minimal_doc,
    text_split,
    download_embedding
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load PDF
extracted_data = load_pdf_files(
    data="C:\\Users\\sjgam\\Downloads\\LLM_AGENT\\LLM_Based_Enterprise_Memory_and_Context_Agent\\data\Medical_book.pdf"
)

# Filtering
filter_data = filter_to_minimal_doc(extracted_data)

# Chunks
text_chunk = text_split(filter_data)

# Vector DB
persist_directory = "db"
embeddings = download_embedding()

vectordb = Chroma.from_documents(
    documents=text_chunk,
    embedding=embeddings,
    persist_directory=persist_directory
)

vectordb.persist()
vectordb = None

# Load vectordb again
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)
