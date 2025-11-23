from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from  langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document

# Extract Text from pdf

def load_pdf_files(data):
    loader = PyPDFLoader(data)   # here data is a SINGLE PDF file
    documents = loader.load()
    return documents

# Filter to minimal
def filter_to_minimal_doc(docs:List[Document])->List[Document]:
    """
    Diven the list of doucument object return the new lis tod Documents object containing 
    only source in the metadata and the orginal page content.
    """
    minimal_docs:List[Document]=[]
    for doc in docs:
        src=doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs

 # Split the document into smaller chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunk=text_splitter.split_documents(minimal_docs)
    return text_chunk

# Download the embedding model
from langchain.embeddings import HuggingFaceEmbeddings
def download_embedding():
    """ 
    Download and return the HuggingFace embedding model.
    """

    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name,
    )
    return embeddings
embeddings=download_embedding()