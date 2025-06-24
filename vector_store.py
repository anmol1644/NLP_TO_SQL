import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import uuid
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Google API key for embeddings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Base directory for vector stores
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", "./vector_stores")


class VectorStoreManager:
    """Manager for vector stores to persist conversation context"""
    
    def __init__(self):
        """Initialize the vector store manager"""
        self.embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-exp-03-07",
            google_api_key=GOOGLE_API_KEY,
        )
        
        # Keep track of active stores to properly close them
        self.active_stores = {}
        
        # Ensure the base directory exists
        os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    def create_store(self, session_id: str) -> str:
        """Create a new vector store for a session"""
        # Create a unique ID for the vector store
        vector_store_id = str(uuid.uuid4())
        
        # Create a directory for this vector store
        store_dir = os.path.join(VECTOR_STORE_DIR, vector_store_id)
        os.makedirs(store_dir, exist_ok=True)
        
        # Initialize an empty vector store
        Chroma(
            persist_directory=store_dir,
            embedding_function=self.embedding_function,
            collection_name=f"session_{session_id}"
        )
        
        return vector_store_id
    
    def get_store(self, vector_store_id: str, session_id: str) -> Optional[Chroma]:
        """Get a vector store by ID"""
        store_dir = os.path.join(VECTOR_STORE_DIR, vector_store_id)
        
        if not os.path.exists(store_dir):
            return None
        
        # Check if we already have this store in memory
        if vector_store_id in self.active_stores:
            return self.active_stores[vector_store_id]
        
        # Create and cache the store
        store = Chroma(
            persist_directory=store_dir,
            embedding_function=self.embedding_function,
            collection_name=f"session_{session_id}"
        )
        
        self.active_stores[vector_store_id] = store
        return store
    
    def add_message_to_store(
        self, 
        vector_store_id: str, 
        session_id: str, 
        content: str, 
        role: str, 
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Add a message to a vector store"""
        vector_store = self.get_store(vector_store_id, session_id)
        
        if not vector_store:
            return False
        
        # Create a document from the message
        meta = metadata or {}
        meta.update({"role": role, "session_id": session_id})
        
        document = Document(
            page_content=content,
            metadata=meta
        )
        
        # Add the document to the vector store
        vector_store.add_documents([document])
        
        return True
    
    def search_context(
        self, 
        vector_store_id: str, 
        session_id: str, 
        query: str, 
        k: int = 5
    ) -> List[Document]:
        """Search for relevant context in a vector store"""
        vector_store = self.get_store(vector_store_id, session_id)
        
        if not vector_store:
            return []
        
        # Search for similar documents
        results = vector_store.similarity_search(query, k=k)
        
        return results
    
    def delete_store(self, vector_store_id: str) -> bool:
        """Delete a vector store"""
        store_dir = os.path.join(VECTOR_STORE_DIR, vector_store_id)
        
        if not os.path.exists(store_dir):
            return False
        
        # Close and remove from active stores if it exists
        if vector_store_id in self.active_stores:
            try:
                store = self.active_stores[vector_store_id]
                # Try to properly close the ChromaDB connection
                if hasattr(store, '_client') and hasattr(store._client, 'close'):
                    store._client.close()
                elif hasattr(store, 'close'):
                    store.close()
            except Exception as e:
                print(f"Warning: Could not properly close vector store {vector_store_id}: {e}")
            
            # Remove from active stores
            del self.active_stores[vector_store_id]
        
        # Delete the directory with retry logic for Windows file locking issues
        import shutil
        import time
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                shutil.rmtree(store_dir)
                return True
            except PermissionError as e:
                if attempt < max_retries - 1:
                    # Wait a bit and try again
                    time.sleep(0.5)
                    continue
                else:
                    # If we can't delete it, just rename it to mark for deletion
                    try:
                        import uuid
                        temp_name = f"{store_dir}_deleted_{uuid.uuid4().hex[:8]}"
                        os.rename(store_dir, temp_name)
                        return True
                    except Exception:
                        # Log the error but don't fail the operation
                        print(f"Warning: Could not delete vector store directory {store_dir}: {e}")
                        return False
            except Exception as e:
                print(f"Error deleting vector store {vector_store_id}: {e}")
                return False
        
        return False


# Singleton instance
vector_store_manager = VectorStoreManager() 