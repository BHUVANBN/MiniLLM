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
import cohere
from pypdf import PdfReader
from tqdm import tqdm
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS

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

# Suppress HTTP request logs from cohere and other libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("cohere").setLevel(logging.WARNING)


class UniversalChatbot:
    """
    Universal Chatbot class that can work with PDF documents or as a general AI assistant.
    """
    
    def __init__(self):
        """Initialize the Universal Chatbot with default configurations."""
        self.current_pdf_path = None
        self.pdf_texts = []
        self.character_split_texts = []
        self.embedding_function = None
        self.vector_db = None
        self.cohere_client = None
        self.pdf_loaded = False
        
        # Configuration
        self.chunk_size = 1000
        self.chunk_overlap = 20
        self.embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
        self.cohere_model = "command-r-08-2024"
        
        logger.info("Universal Chatbot initialized")
    
    def setup_api_keys(self) -> bool:
        """
        Setup API keys for HuggingFace and Cohere.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            # HuggingFace API Key
            hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
            if not hf_api_key:
                print("Please enter your HuggingFace API key:")
                hf_api_key = getpass("HuggingFace API Key: ")
                os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
            
            # Cohere API Key
            cohere_api_key = os.getenv("COHERE_API_KEY")
            if not cohere_api_key:
                print("Please enter your Cohere API key:")
                cohere_api_key = getpass("Cohere API Key: ")
            
            # Initialize Cohere client
            self.cohere_client = cohere.Client(cohere_api_key)
            
            # Initialize embedding function
            try:
                self.embedding_function = HuggingFaceInferenceAPIEmbeddings(
                    api_key=hf_api_key,
                    model_name=self.embedding_model
                )
                logger.info(f"Embedding function initialized with model: {self.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to initialize HuggingFace Inference API: {e}")
                logger.info("Trying alternative local embedding approach...")
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    self.embedding_function = HuggingFaceEmbeddings(
                        model_name=self.embedding_model,
                        model_kwargs={'device': 'cpu'}
                    )
                    logger.info(f"Local embedding function initialized with model: {self.embedding_model}")
                except Exception as e2:
                    logger.error(f"Failed to initialize local embeddings: {e2}")
                    return False
            
            logger.info("API keys setup successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up API keys: {e}")
            return False
    
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
            
            reader = PdfReader(pdf_path)
            self.pdf_texts = [p.extract_text().strip() for p in reader.pages]
            
            # Filter out empty strings
            self.pdf_texts = [text for text in self.pdf_texts if text]
            
            logger.info(f"Successfully loaded {len(self.pdf_texts)} pages from PDF")
            return True
            
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return False
    
    def create_text_chunks(self) -> bool:
        """
        Split PDF text into chunks for processing.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.pdf_texts:
                logger.error("No PDF texts loaded. Please load PDF first.")
                return False
            
            logger.info("Creating text chunks...")
            
            character_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            self.character_split_texts = character_splitter.split_text('\n\n'.join(self.pdf_texts))
            
            logger.info(f"Created {len(self.character_split_texts)} text chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating text chunks: {e}")
            return False
    
    def create_vector_database(self) -> bool:
        """
        Create FAISS vector database from text chunks.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.character_split_texts:
                logger.error("No text chunks available. Please create chunks first.")
                return False
            
            if not self.embedding_function:
                logger.error("Embedding function not initialized. Please setup API keys first.")
                return False
            
            logger.info(f"Creating vector database from {len(self.character_split_texts)} text chunks...")
            
            # Filter out empty chunks
            valid_chunks = [chunk for chunk in self.character_split_texts if chunk.strip()]
            
            if not valid_chunks:
                logger.error("No valid text chunks found after filtering.")
                return False
            
            logger.info(f"Using {len(valid_chunks)} valid chunks for vector database...")
            
            # Test embedding function with a small sample first
            try:
                logger.info("Testing embedding function...")
                test_embedding = self.embedding_function.embed_query("test")
                logger.info(f"Embedding test successful, dimension: {len(test_embedding)}")
            except Exception as embed_error:
                logger.error(f"Embedding function test failed: {embed_error}")
                logger.info("Switching to local embedding fallback...")
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    self.embedding_function = HuggingFaceEmbeddings(
                        model_name=self.embedding_model,
                        model_kwargs={'device': 'cpu'}
                    )
                    logger.info("Local embedding function initialized successfully")
                    # Test the local embedding
                    test_embedding = self.embedding_function.embed_query("test")
                    logger.info(f"Local embedding test successful, dimension: {len(test_embedding)}")
                except Exception as local_error:
                    logger.error(f"Local embedding fallback failed: {local_error}")
                    return False
            
            # Create vector database with error handling
            logger.info("Creating FAISS vector database...")
            self.vector_db = FAISS.from_texts(valid_chunks, self.embedding_function)
            self.pdf_loaded = True
            
            logger.info(f"Vector database created successfully with {self.vector_db.index.ntotal} documents")
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
        self.pdf_texts = []
        self.character_split_texts = []
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
            information = "\n\n".join([doc.page_content for doc in retrieved_documents])
            
            # Create system prompt for PDF context
            system_prompt = f"""You are a helpful AI assistant that answers questions based on the content of a PDF document.
The PDF file is: {Path(self.current_pdf_path).name if self.current_pdf_path else 'Unknown'}

You will be shown the user's question and relevant information from the document.
Answer the user's question using only the information provided from the document.
If the information is not sufficient to answer the question, say so clearly."""
            
            # Try different models in order of preference (updated for current availability)
            models_to_try = ["command-r-08-2024", "command-r-03-2024", "command"]
            
            for model in models_to_try:
                try:
                    # Generate response using Cohere
                    response = self.cohere_client.chat(
                        model=model,
                        message=f"Question: {query}\n\nDocument Information: {information}",
                        preamble=system_prompt
                    )
                    
                    # If successful, update the default model and break
                    self.cohere_model = model
                    break
                    
                except Exception as model_error:
                    logger.warning(f"Model {model} failed: {model_error}")
                    if model == models_to_try[-1]:  # Last model
                        return "All Cohere models are currently unavailable. Please try again later.", ""
                    continue
            
            # Combine source texts
            source_text = "\n\n".join([doc.page_content for doc in retrieved_documents])
            
            return response.text, source_text
            
        except Exception as e:
            logger.error(f"Error in PDF chat: {e}")
            return f"Error: {str(e)}", ""
    
    def chat_general(self, query: str) -> str:
        """
        General chat without PDF context.
        
        Args:
            query (str): User's question
            
        Returns:
            str: AI response
        """
        try:
            if not self.cohere_client:
                return "Cohere client not initialized. Please setup API keys first."
            
            if not query.strip():
                return "Please enter a valid question."
            
            # Create system prompt for general chat
            system_prompt = """You are a helpful, knowledgeable, and friendly AI assistant. 
You can help with a wide variety of topics including answering questions, providing explanations, 
helping with tasks, and having conversations. Be informative, accurate, and helpful in your responses."""
            
            # Try different models in order of preference (updated for current availability)
            models_to_try = ["command-r-08-2024", "command-r-03-2024", "command"]
            
            for model in models_to_try:
                try:
                    # Generate response using Cohere
                    response = self.cohere_client.chat(
                        model=model,
                        message=query,
                        preamble=system_prompt
                    )
                    
                    # If successful, update the default model
                    self.cohere_model = model
                    return response.text
                    
                except Exception as model_error:
                    logger.warning(f"Model {model} failed: {model_error}")
                    continue
            
            # If all models fail
            return "All Cohere models are currently unavailable. Please try again later."
            
        except Exception as e:
            logger.error(f"Error in general chat: {e}")
            return f"Error: {str(e)}"
    
    def get_pdf_info(self) -> str:
        """
        Get information about the currently loaded PDF.
        
        Returns:
            str: PDF information
        """
        if not self.pdf_loaded:
            return "No PDF currently loaded."
        
        return f"""
📄 Current PDF: {Path(self.current_pdf_path).name if self.current_pdf_path else 'Unknown'}
📊 Pages: {len(self.pdf_texts)}
📝 Text Chunks: {len(self.character_split_texts)}
🔍 Vector Database: {self.vector_db.index.ntotal if self.vector_db else 0} documents
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
