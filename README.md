# 🏥 Medical Chatbot

An intelligent medical chatbot powered by LLM (Llama 3.1) and RAG (Retrieval-Augmented Generation) architecture. This chatbot can answer medical questions by retrieving relevant information from medical documents stored in a Pinecone vector database.

## 📋 Features

- **RAG Architecture**: Combines retrieval and generation for accurate medical responses
- **Vector Search**: Uses Pinecone for efficient similarity search across medical documents
- **Local LLM**: Powered by Llama 3.1 running locally via Ollama
- **Web Interface**: Flask-based chat interface for easy interaction
- **Document Processing**: Automatically processes and indexes PDF medical documents

## 🛠️ Tech Stack

- **Python** - Core programming language
- **LangChain** - LLM framework for building the RAG pipeline
- **Flask** - Web framework for the chat interface
- **Ollama (Llama 3.1)** - Local LLM for response generation
- **Pinecone** - Vector database for document embeddings
- **HuggingFace Embeddings** - Sentence transformers for creating embeddings
- **PyPDF** - PDF document processing

## 📁 Project Structure

```
Medical-Chatbot/
├── app.py                  # Flask application (main chatbot interface)
├── store_index.py          # Script to create and populate Pinecone index
├── setup.py                # Package setup configuration
├── requirements.txt        # Python dependencies
├── data/                   # Directory for medical PDF documents
├── src/
│   ├── helper.py          # Helper functions (PDF loading, embeddings)
│   └── prompt.py          # Prompt templates for the LLM
├── static/
│   └── style.css          # CSS styling for web interface
└── templates/
    └── chat.html          # HTML template for chat interface
```

## 🚀 Setup Instructions

### Prerequisites

1. **Install Ollama and download Llama 3.1**:
   ```bash
   # Visit https://ollama.ai and install Ollama
   # Then pull the Llama 3.1 model
   ollama pull llama3.1:latest
   ```

2. **Get Pinecone API Key**:
   - Sign up at [Pinecone](https://www.pinecone.io/)
   - Create a new API key from your dashboard

### Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Medical-Chatbot
   ```

2. **Create a conda environment**:
   ```bash
   conda create -n mchatbot python=3.8 -y
   conda activate mchatbot
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your Pinecone API key:
   ```
   PINECONE_API_KEY=your_api_key_here
   ```

5. **Add medical documents**:
   Place your medical PDF documents in the `data/` directory.

## 🏃‍♂️ Usage

### Step 1: Create and Populate the Vector Database

Run this **once** to index your medical documents:

```bash
python store_index.py
```

This script will:
- Load PDF documents from the `data/` directory
- Split them into chunks
- Create embeddings using HuggingFace
- Store them in Pinecone vector database

### Step 2: Start the Chatbot

```bash
python app.py
```

<<<<<<< HEAD
The Flask application will start on `http://127.0.0.1:5000/`

Open your browser and navigate to the URL to start chatting!

## 💡 How It Works

1. **Document Indexing** ([store_index.py](store_index.py)):
   - Loads medical PDFs from the `data/` folder
   - Splits documents into manageable chunks
   - Generates embeddings using HuggingFace Sentence Transformers (384 dimensions)
   - Stores embeddings in Pinecone with cosine similarity metric

2. **Query Processing** ([app.py](app.py)):
   - User asks a medical question through the web interface
   - Question is converted to an embedding
   - Top 3 most similar document chunks are retrieved from Pinecone
   - Retrieved context + question are sent to Llama 3.1
   - LLM generates a response based on the retrieved medical information

## 🔧 Configuration

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions)
- **LLM Model**: `llama3.1:latest` (via Ollama)
- **Vector Database**: Pinecone (AWS us-east-1, Serverless)
- **Retrieval**: Top 3 similar documents (similarity search)
- **Temperature**: 0.0 (deterministic responses)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## ⚠️ Disclaimer

This chatbot is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.

## 📧 Contact

For questions or feedback, please open an issue in this repository.