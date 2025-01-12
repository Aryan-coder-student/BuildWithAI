import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.prompts import ChatPromptTemplate
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Chatbot",
    page_icon="🤖",
    layout="wide"
)
directory_path = "Papers"
# Add title and description
st.title("Deep Learning revolution Paper Chatbot 🤖")
st.markdown("""
This chatbot can answer questions about the Generative Adversarial Nets , Attention All You Need and Autoencoderes papers (.
""")

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.retriever = None
    st.session_state.llm = None

def initialize_chatbot():
    """Initialize the chatbot with the PDF dataset"""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        llm = ChatGroq(temperature=0.8, model="llama-3.3-70b-specdec", api_key=api_key)
        persist_directory = "chroma_db"
        if not os.path.exists(persist_directory):
            loader = DirectoryLoader(directory_path, glob="**/*.pdf") 
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(documents)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            db.persist()
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
        
        retriever = db.as_retriever(k=min(len(db), 15))
        st.session_state.retriever = retriever
        st.session_state.llm = llm
        st.session_state.initialized = True
        
        return True
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return False

def generate_response(query):
    """Generate response using the chatbot"""
    try:
        prompt = ChatPromptTemplate.from_template("""
        Answer the following question based only on the provided context.
        Think step by step before providing a detailed answer.
        <context>
        {context}
        </context>
        Question: {input}
        """)
        
        chain = create_stuff_documents_chain(llm=st.session_state.llm, prompt=prompt)
        retriever_chain = create_retrieval_chain(st.session_state.retriever, chain)
        response = retriever_chain.invoke({"input": query})
        return response["answer"]
    except Exception as e:
        return f"An error occurred: {str(e)}"
if not st.session_state.initialized:
    with st.spinner("Initializing chatbot..."):
        if initialize_chatbot():
            st.success("Chatbot initialized successfully!")
        else:
            st.error("Failed to initialize chatbot. Please check if the PDF file exists and try again.")
if st.session_state.initialized:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Ask a question about the YOLO paper"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
with st.sidebar:
    st.title("About")
    st.markdown("""
    This chatbot uses:
    - Groq LLM (llama-3.3-70b)
    - LangChain for document processing
    - Streamlit for the web interface
    - HuggingFace embeddings
    - Dataset: Generative Adversarial Nets (1406.2661v1) , Attention All You Need (1706.03762v7) , Autoencoderes (2003.05991v2) papers 
    
    Ask questions about the YOLO paper's contents!
    """)

    # Add reset button to clear the vector store
    if st.button("Reset Vector Store"):
        import shutil
        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")
            st.session_state.initialized = False
            st.experimental_rerun()
