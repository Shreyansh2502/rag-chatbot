# Support Assistant - RAG-based Chatbot

A Retrieval-Augmented Generation (RAG) chatbot trained on customer support documentation to assist users by answering queries and providing relevant support information.

## Features

- Answers questions based on provided documentation
- Supports PDF and DOCX file uploads
- Modern, user-friendly chat interface
- Real-time response with typing indicators
- Clear "I don't know" responses for out-of-scope questions

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the server:
```bash
uvicorn chatbot:app --host 0.0.0.0 --port 8199
```

4. Access the chatbot:
Open your browser and navigate to `http://localhost:8199`

## Usage

1. **Chat Interface**
   - Type your question in the input box
   - Press Enter or click Send to submit
   - The chatbot will respond based on the available documentation

2. **Document Upload**
   - Click the "Upload Document" button
   - Select a PDF or DOCX file
   - The chatbot will process and incorporate the document into its knowledge base

3. **Response Types**
   - Direct answers from the documentation
   - "I don't know" for questions outside the knowledge base
   - Error messages for technical issues

## Technical Details

- Built with FastAPI for the backend
- Uses ChromaDB for vector storage
- Implements sentence-transformers for embeddings
- Supports PDF and DOCX file processing
- Modern UI with real-time updates

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- ChromaDB
- Sentence Transformers
- PyPDF2
- python-docx
- BeautifulSoup4
- python-multipart

## License

MIT License 