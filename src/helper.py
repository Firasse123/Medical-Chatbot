from langchain.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


#extract data from pdf
def load_pdf(data):
    loader=DirectoryLoader(data,
                    glob="*.pdf",
                    loader_cls=PyMuPDFLoader)
    documents=loader.load()
    return documents

#➡️ "Go to the folder data
#➡️ find all .pdf files
#➡️ load them using PyMuPDFLoader
#➡️ prepare them for splitting and embedding"



#split data into chunks
def split_data(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks



#download embedding model 
def download_hugging_face_embedding():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings