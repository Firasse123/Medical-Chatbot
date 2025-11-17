from src.helper import load_pdf, split_data,download_hugging_face_embedding
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
load_dotenv()
import os

Pinecone_api_key=os.getenv("PINECONE_API_KEY")
index_name="medical-chatbot"


pc=Pinecone(api_key=Pinecone_api_key)
if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
extracted_data=load_pdf("data/")
text_chunks=split_data(extracted_data)
embeddings=download_hugging_face_embedding()




docsearch=PineconeVectorStore.from_documents(text_chunks,embeddings,index_name=index_name)