#!/usr/bin/env python3
"""
Universal PDF Chatbot Application

This application creates a flexible chatbot that can work with any PDF document
or function as a general-purpose AI assistant without PDF context.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from getpass import getpass

import numpy as np
from pypdf import PdfReader
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('universal_chatbot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress HTTP request logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class UniversalChatbot:
    """
    Universal Chatbot class that can work with PDF documents or as a general AI assistant.
    """
    
    def __init__(self):
        """Initialize the Universal Chatbot with default configurations."""
        self.current_pdf_path = None
        self.documents = [] # List of Document objects
        self.character_split_docs = [] # List of Document objects
        self.embedding_function = None
        self.vector_db = None
        self.pdf_loaded = False
        
        # Configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.embedding_model = "all-MiniLM-L6-v2"
        self.ollama_model = "llama3.2:latest" # Default Ollama model
        self.vector_db_path = "./data/vector_store"
        
        # Local LLM and Embeddings
        self.llm = None
        
        logger.info("Universal Chatbot initialized")
    
    def setup_local_models(self, model_name: str = "llama3.2:latest") -> bool:
        """
        Setup local models (Ollama and SentenceTransformers).
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            self.ollama_model = model_name
            logger.info(f"Setting up local Ollama LLM with model: {self.ollama_model}")
            
            # Initialize Ollama
            self.llm = OllamaLLM(model=self.ollama_model)
            
            # Initialize embedding function (local)
            logger.info(f"Initializing local embeddings with model: {self.embedding_model}")
            self.embedding_function = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'}
            )
            
            logger.info("Local models setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up local models: {e}")
            return False

    def setup_api_keys(self) -> bool:
        """
        Setup API keys for Cohere (optional fallback).
        """
        # This can be kept as an optional fallback
        return self.setup_local_models() 
    
    def load_pdf(self, pdf_path: str) -> bool:
        """
        Load and extract text from PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not Path(pdf_path).exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            self.current_pdf_path = pdf_path
            logger.info(f"Loading PDF: {pdf_path}")
            
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(pdf_path)
            self.documents = loader.load()
            
            # Add metadata
            for doc in self.documents:
                doc.metadata['source_file'] = Path(pdf_path).name
                doc.metadata['file_type'] = 'pdf'
            
            logger.info(f"Successfully loaded {len(self.documents)} pages from PDF")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return False

    def load_directory(self, directory_path: str) -> bool:
        """
        Process all PDF files in a directory.
        """
        try:
            pdf_dir = Path(directory_path)
            if not pdf_dir.exists() or not pdf_dir.is_dir():
                logger.error(f"Directory not found: {directory_path}")
                return False
                
            pdf_files = list(pdf_dir.glob("**/*.pdf"))
            logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            self.documents = []
            from langchain_community.document_loaders import PyPDFLoader
            
            for pdf_file in pdf_files:
                try:
                    loader = PyPDFLoader(str(pdf_file))
                    docs = loader.load()
                    # Add metadata
                    for doc in docs:
                        doc.metadata['source_file'] = pdf_file.name
                        doc.metadata['file_type'] = 'pdf'
                    self.documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to load {pdf_file}: {e}")
            
            self.current_pdf_path = directory_path
            logger.info(f"Successfully loaded {len(self.documents)} pages from directory")
            return True
            
        except Exception as e:
            logger.error(f"Error loading directory: {e}")
            return False
    
    def create_text_chunks(self) -> bool:
        """
        Split PDF documents into chunks for processing.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.documents:
                logger.error("No documents loaded. Please load PDF/directory first.")
                return False
            
            logger.info("Creating text chunks...")
            
            character_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            self.character_split_docs = character_splitter.split_documents(self.documents)
            
            logger.info(f"Created {len(self.character_split_docs)} text chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating text chunks: {e}")
            return False
    
    def create_vector_database(self) -> bool:
        """
        Create ChromaDB vector database from text chunks.
        """
        try:
            if not self.character_split_docs:
                logger.error("No text chunks available.")
                return False
            
            if not self.embedding_function:
                logger.error("Embedding function not initialized.")
                return False
            
            logger.info(f"Creating vector database from {len(self.character_split_docs)} text chunks...")
            
            # Create vector database using Chroma
            if not os.path.exists(os.path.dirname(self.vector_db_path)):
                os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)

            self.vector_db = Chroma.from_documents(
                documents=self.character_split_docs,
                embedding=self.embedding_function,
                persist_directory=self.vector_db_path
            )
            
            self.pdf_loaded = True
            logger.info("Vector database created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating vector database: {type(e).__name__}: {str(e)}")
            return False
    
    def process_pdf(self, pdf_path: str) -> bool:
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.load_pdf(pdf_path):
            return False
        
        if not self.create_text_chunks():
            return False
        
        if not self.create_vector_database():
            return False
        
        return True
    
    def clear_pdf_context(self):
        """Clear the current PDF context."""
        self.current_pdf_path = None
        self.documents = []
        self.character_split_docs = []
        self.vector_db = None
        self.pdf_loaded = False
        logger.info("PDF context cleared")
    
    def chat_with_pdf(self, query: str) -> Tuple[str, str]:
        """
        Chat with the loaded PDF using RAG.
        
        Args:
            query (str): User's question
            
        Returns:
            Tuple[str, str]: Response and source text
        """
        try:
            if not self.pdf_loaded or not self.vector_db:
                return "No PDF loaded. Please load a PDF first.", ""
            
            if not query.strip():
                return "Please enter a valid question.", ""
            
            # Retrieve relevant documents
            retrieved_documents = self.vector_db.similarity_search(query, k=5)
            
            if not retrieved_documents:
                return "No relevant information found for your query in the PDF.", ""
            
            # Extract content from retrieved documents
            context = "\n\n".join([doc.page_content for doc in retrieved_documents])
            
            # Create system prompt for PDF context
            prompt = f"""You are a helpful AI assistant that answers questions based on the content of a PDF document.
Document Information:
{context}

Question: {query}

Answer the user's question using ONLY the information provided above. If the information is not sufficient to answer the question, say so clearly."""
            
            # Generate response using Ollama
            response = self.llm.invoke(prompt)
            
            return response, context
            
        except Exception as e:
            logger.error(f"Error in PDF chat: {e}")
            return f"Error: {str(e)}", ""
    
    def chat_general(self, query: str) -> str:
        """
        General chat using Ollama.
        """
        try:
            if not self.llm:
                return "Model not initialized."
            
            if not query.strip():
                return "Please enter a valid question."
            
            # Generate response using Ollama
            response = self.llm.invoke(query)
            return response
            
        except Exception as e:
            logger.error(f"Error in general chat: {e}")
            return f"Error: {str(e)}"
    
    def get_pdf_info(self) -> str:
        """
        Get information about the currently loaded content.
        """
        if not self.pdf_loaded:
            return "No content currently loaded."
        
        return f"""
📑 Source: {Path(self.current_pdf_path).name if self.current_pdf_path else 'Unknown'}
📊 Pages: {len(self.documents)}
📝 Text Chunks: {len(self.character_split_docs)}
🔍 Vector Database: ChromaDB (Offline)
"""
    
    def initialize_apis(self) -> bool:
        """
        Initialize API connections without PDF.
        
        Returns:
            bool: True if successful, False otherwise
        """
        return self.setup_api_keys()


def main():
    """Main function for testing the Universal Chatbot."""
    chatbot = UniversalChatbot()
    
    if not chatbot.initialize_apis():
        print("Failed to initialize APIs")
        return
    
    print("Universal Chatbot initialized successfully!")
    print("You can now use it for general chat or load a PDF for document-specific queries.")


if __name__ == "__main__":
    main()
