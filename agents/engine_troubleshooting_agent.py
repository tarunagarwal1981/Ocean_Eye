import logging
from typing import List, Dict, Tuple, Any
import openai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition
import os
import streamlit as st
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EngineTroubleshootingRAG:
    """Handles engine troubleshooting queries using RAG approach."""
    
    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.qdrant_client = self._init_qdrant_client()
        openai.api_key = self._get_api_key()
        
        # Check vector store during initialization
        if self.is_initialized():
            self._check_vector_store()
    
    def _check_vector_store(self):
        """Diagnostic function to check vector store"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections()
            logger.info(f"Available collections: {collections}")
            
            # Get collection info
            collection_info = self.qdrant_client.get_collection('manual_vectors')
            logger.info(f"Collection info: {collection_info}")
            
            # Get a sample of vectors
            sample = self.qdrant_client.scroll(
                collection_name="manual_vectors",
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            if sample[0]:  # if there are any vectors
                logger.info("Sample vector payloads:")
                for point in sample[0]:
                    logger.info(f"Point ID: {point.id}")
                    logger.info(f"Payload: {point.payload}")
            else:
                logger.warning("No vectors found in collection")
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking vector store: {e}")
            return False

    def is_initialized(self) -> bool:
        """Check if critical components are initialized"""
        return all([
            self.embedding_model is not None,
            self.qdrant_client is not None,
            openai.api_key is not None
        ])
        
    def _load_embedding_model(self):
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return None

    def _init_qdrant_client(self):
        try:
            client = QdrantClient(
                url=st.secrets["qdrant"]["url"],
                api_key=st.secrets["qdrant"]["api_key"],
                timeout=30
            )
            # Test connection
            client.get_collections()
            return client
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            return None

    def _get_api_key(self):
        try:
            if 'openai' in st.secrets:
                return st.secrets['openai']['api_key']
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.error("OpenAI API key not found")
                return None
            return api_key
        except Exception as e:
            logger.error(f"Error getting OpenAI API key: {e}")
            return None

    def _search_qdrant(self, query_vector: List[float], max_retries: int = 3) -> List[Any]:
        """Search Qdrant with retries"""
        last_error = None
        for attempt in range(max_retries):
            try:
                # Try searching without type filter first
                results = self.qdrant_client.search(
                    collection_name="manual_vectors",
                    query_vector=query_vector,
                    limit=10
                )
                
                if results:
                    logger.info(f"Found {len(results)} results")
                    return results
                    
                time.sleep(1)  # Small delay between retries
                
            except Exception as e:
                last_error = e
                logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
        
        logger.error(f"All Qdrant search attempts failed: {last_error}")
        return []

    def process_engine_query(self, 
                           question: str, 
                           vessel_name: str = None, 
                           chat_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        """Process an engine-related query"""
        if not self.is_initialized():
            return ("System initialization error. Please try again later.", [])

        try:
            # Get embedding for the question
            query_embedding = self.embedding_model.encode(question).tolist()
            
            # Search Qdrant
            search_results = self._search_qdrant(query_embedding)
            if not search_results:
                return ("No relevant engine documentation found. Please try a different query.", [])

            # Process results
            context = ""
            relevant_images = []
            
            for result in search_results:
                try:
                    payload = result.payload
                    if payload.get("type") == "text":
                        context += f"From {payload.get('file_name', 'unknown')}, page {payload.get('page', 'unknown')}:\n"
                        context += f"{payload.get('content', '')}\n"
                    elif payload.get("type") == "image":
                        relevant_images.append(payload)
                        context += f"\nTechnical diagram: {payload.get('image_name', 'unknown')}"
                        context += f" from {payload.get('file_name', 'unknown')}, page {payload.get('page', 'unknown')}:\n"
                        context += f"Diagram context: {payload.get('surrounding_text', '')}\n"
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue

            if not context:
                return ("Could not process the engine documentation effectively.", [])

            # Prepare GPT messages
            messages = [
                {"role": "system", "content": """You are an expert marine engineer 
                specializing in engine troubleshooting. When answering:
                1. Always start with safety considerations if applicable
                2. Provide step-by-step troubleshooting procedures
                3. Reference specific engine components and systems accurately
                4. Cite relevant technical diagrams when available
                5. Mention any required tools or equipment
                6. Include relevant maintenance schedule considerations
                7. Highlight potential risks or complications"""}
            ]
            
            if chat_history:
                messages.extend(chat_history)
                
            vessel_context = f" for vessel {vessel_name}" if vessel_name else ""
            
            messages.extend([
                {"role": "user", "content": 
                 f"Context:\n{context}\n\nQuestion about engine troubleshooting{vessel_context}: {question}"}
            ])

            # Get GPT response
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7
                )
                answer = response.choices[0].message['content'].strip()
                return answer, relevant_images
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                return ("Error generating response. Please try again.", relevant_images)

        except Exception as e:
            logger.error(f"Error processing engine query: {e}")
            return "An error occurred while processing your query. Please try again.", []

def analyze_engine_troubleshooting(vessel_name: str, query: str) -> Tuple[str, List[Dict]]:
    """Main function to analyze engine troubleshooting queries."""
    try:
        rag = EngineTroubleshootingRAG()
        if not rag.is_initialized():
            return ("Engine troubleshooting system is not properly initialized. Please try again later.", [])
        return rag.process_engine_query(query, vessel_name)
    except Exception as e:
        logger.error(f"Error in engine troubleshooting analysis: {e}")
        return "Error processing engine troubleshooting query.", []
