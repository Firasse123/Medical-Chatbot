from platform import system
from flask import Flask, render_template, jsonify,request
from src.helper import download_hugging_face_embedding
from src.prompt import prompt_template
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.prompts import PromptTemplate
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
app=Flask(__name__)

Pinecone_API_Key=os.getenv("PINECONE_API_KEY")
index_name="medical-chatbot"
embeddings=download_hugging_face_embedding()


docsearch=PineconeVectorStore.from_existing_index(index_name,embeddings)


prompt=ChatPromptTemplate.from_messages(
    [("system",prompt_template),
     ("user","{input}")]
)
llm = ChatOllama(
    model="llama3.1:latest",
    temperature=0.0
)

retriever=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})

question_answer_chain=create_stuff_documents_chain(llm,prompt)
chain=create_retrieval_chain(retriever,question_answer_chain)
                             
@app.route("/")
def index():
    return render_template("chat.html")  
@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    result = chain.invoke({"input": input})
    return str(result["answer"])

                    

if __name__ == '__main__':
    app.run(debug=True)
