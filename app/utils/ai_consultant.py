"""
AI Microstructure Consultant for MatSAM.
Provides RAG-based advice on SEM image enhancement and parameter optimization.
"""

import os
import sys
from pathlib import Path
import json
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIConsultant")

class AIConsultant:
    """
    Expert AI Consultant for SEM Microstructure Analysis.
    Supports RAG (Retrieval-Augmented Generation) from research papers.
    """
    
    def __init__(self, provider="ollama", model="llama3", api_key=None, base_url=None):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.knowledge_base_path = Path(__file__).parent.parent / "knowledge_base"
        self.vector_db = None
        self.is_initialized = False
        
    def initialize_rag(self) -> tuple[bool, str]:
        """
        Initialize the RAG system by indexing papers in the knowledge_base folder.
        Returns: (success, message)
        """
        try:
            from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            
            logger.info(f"Initializing RAG from {self.knowledge_base_path}")
            
            if not self.knowledge_base_path.exists():
                msg = f"Knowledge base directory {self.knowledge_base_path} not found."
                logger.warning(msg)
                return False, msg
                
            # Load PDFs
            loader = DirectoryLoader(
                str(self.knowledge_base_path),
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            
            if not documents:
                msg = "No PDF documents found in knowledge base. Please add .pdf files to app/knowledge_base/"
                logger.warning(msg)
                return False, msg
                
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(documents)
            
            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_db = FAISS.from_documents(splits, embeddings)
            
            self.is_initialized = True
            msg = "RAG system initialized successfully with research papers."
            logger.info(msg)
            return True, msg
            
        except ImportError as e:
            missing_pkg = str(e).split("'")[-2] if "'" in str(e) else str(e)
            msg = f"Missing dependency: {missing_pkg}. Please run: pip install -r requirements_ai.txt"
            logger.error(msg)
            return False, msg
        except Exception as e:
            msg = f"Error initializing RAG: {str(e)}"
            logger.error(msg)
            return False, msg

    def query(self, question: str, context: Dict[str, Any] = None) -> str:
        """
        Query the AI consultant with a question and optional image/parameter context.
        """
        # 1. Retrieve relevant info from RAG
        rag_context = ""
        if self.is_initialized and self.vector_db:
            docs = self.vector_db.similarity_search(question, k=3)
            rag_context = "\n\nRelevant research excerpts:\n" + "\n---\n".join([doc.page_content for doc in docs])
        
        # 2. Format system prompt with tool expertise
        system_prompt = self._get_system_prompt(context)
        
        # 3. Call LLM
        full_prompt = f"{system_prompt}\n\nContext from Research Papers:\n{rag_context}\n\nUser Question: {question}"
        
        return self._call_llm(full_prompt)

    def _get_system_prompt(self, context: Dict[str, Any] = None) -> str:
        """Get the core expertise instructions for the AI."""
        base_prompt = """You are the MatSAM Expert Consultant, a specialized AI for Scanning Electron Microscopy (SEM) image analysis.
Your expertise is in optimizing preprocessing pipelines to enhance linear features (like grain boundaries) in microstructure micrographs.

TECHNICAL SCHEMA:
We use a 'SEMPreprocessor' with the following parameters:
- Frangi Scales: sigmas for Hessian-based line detection (e.g., [0.3, 0.7, 1.5]).
- DoG Sigmas: Difference-of-Gaussians (sigma_small: 0.1-5, sigma_large: 1-25) to highlight edges.
- CLAHE: Local contrast enhancement (clip_limit: 1-30).
- Dirt Threshold: Removes bright artifacts (0.01-1.0).
- Blending: Blends original with enhanced map (0-1.5).

YOUR MISSION:
1. Ground your advice in the provided research paper excerpts.
2. If given image metrics (jaggedness, coverage), use them to diagnose problems.
3. Suggest specific numerical slider adjustments.
4. Explain WHY based on image processing theory (e.g., 'Increasing sigma_large helps remove background shading').

CONCISE STYLE: Provide actionable recommendations first, then technical reasoning.
"""
        if context:
            # Use default=str to handle non-serializable types like numpy bools or floats
            context_str = "\nCURRENT IMAGE STATE:\n" + json.dumps(context, indent=2, default=str)
            return base_prompt + context_str
        return base_prompt

    def _call_llm(self, prompt: str) -> str:
        """Call the selected LLM provider."""
        if self.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.provider in ["openai", "gemini"]:
            return self._call_cloud_api(prompt)
        else:
            return "Error: Unknown AI provider configuration."

    def _call_ollama(self, prompt: str) -> str:
        """Call local Ollama instance."""
        try:
            import requests
            url = f"{self.base_url or 'http://localhost:11434'}/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            return response.json().get("response", "No response from local AI.")
        except Exception as e:
            return f"Ollama Error: {e}. Ensure Ollama is running at {url}"

    def _call_cloud_api(self, prompt: str) -> str:
        """Call Cloud API (OpenAI/Gemini)."""
        # Simple placeholder for Cloud API integration
        # Real implementation would use openai or google-generativeai packages
        return "Not implemented: Please configure OpenAI/Gemini API key in Sidebar."

def get_consultant_instance(st_session_state):
    """Helper for Streamlit state management."""
    if "ai_consultant" not in st_session_state:
        st_session_state.ai_consultant = AIConsultant()
    return st_session_state.ai_consultant
