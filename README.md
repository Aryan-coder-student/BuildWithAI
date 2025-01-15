# AI Research Papers Chatbot 🤖
App Link :-  https://ragchatbotbuildwithai.streamlit.app/ 
An intelligent chatbot powered by Groq LLM that can answer questions about fundamental AI research papers, including GANs, Transformers, and Autoencoders.

![image](https://github.com/user-attachments/assets/dcbf9d63-8103-4203-9507-4988e8cc61da)

## 🎯 Features

- Interactive chat interface built with Streamlit
- Powered by Groq's llama-3.3-70b model for high-quality responses
- Semantic search using HuggingFace embeddings
- Vector storage with ChromaDB for efficient retrieval
- Supports multiple research papers simultaneously
- Easy-to-use web interface
- Real-time response generation

## 📚 Included Research Papers

The chatbot is currently trained on these seminal AI papers:
- Generative Adversarial Nets (1406.2661v1)
- Attention Is All You Need (1706.03762v7)
- Autoencoders (2003.05991v2)

## 🛠️ Installation

1. Clone the repository:
```bash
https://github.com/Aryan-coder-student/BuildWithAI.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

4. Place your PDF papers in the `Papers` directory.

## 🚀 Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

3. Start asking questions about the research papers!

## 🔧 Technical Architecture

- **LLM**: Groq's llama-3.3-70b model for generating responses
- **Embeddings**: HuggingFace's sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: ChromaDB for efficient document retrieval
- **Frontend**: Streamlit for the web interface
- **Document Processing**: LangChain for PDF processing and chunking

## 📝 Environment Variables

The following environment variables are required:

```env
GROQ_API_KEY=your_groq_api_key
```

## 🔄 Reset Vector Store

You can reset the vector store anytime using the "Reset Vector Store" button in the sidebar. This will:
- Delete the existing ChromaDB database
- Reinitialize the system with fresh embeddings
- Rebuild the vector store from scratch

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Groq team for providing the LLM API
- LangChain for the document processing framework
- Streamlit for the web application framework
- The authors of the original research papers


---
Made with ❤️ by [Aryan Pahari]
