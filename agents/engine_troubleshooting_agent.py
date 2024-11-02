import logging
from typing import List, Dict, Tuple, Any, Optional
import openai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Filter, FieldCondition, Range
import os
import nltk
from transformers import pipeline
import streamlit as st
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_nltk():
    """Initialize NLTK data with proper error handling"""
    try:
        # Create the NLTK data directory if it doesn't exist
        nltk_data_dir = '/tmp/nltk_data'
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)

        # Download required NLTK data with retries
        resources = ['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
        max_retries = 3
        
        for resource in resources:
            for attempt in range(max_retries):
                try:
                    nltk.download(resource, quiet=True, download_dir=nltk_data_dir)
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.warning(f"Failed to download NLTK resource {resource}: {e}")
                    else:
                        time.sleep(1)  # Wait before retrying
        
        return True
    except Exception as e:
        logger.error(f"Failed to setup NLTK: {e}")
        return False

class EngineTextProcessor:
    """Processes engine-related text with specialized technical understanding."""
    
    def __init__(self, ner_model):
        self.ner_model = ner_model
        self.nltk_initialized = setup_nltk()
        
    def process_text(self, text: str) -> Dict[str, Any]:
        try:
            # Simple sentence splitting if NLTK fails
            if self.nltk_initialized:
                try:
                    sentences = nltk.sent_tokenize(text)
                    tokens = nltk.word_tokenize(text)
                    tagged = nltk.pos_tag(tokens)
                except Exception:
                    sentences = [s.strip() for s in text.split('.') if s.strip()]
                    tokens = text.split()
                    tagged = [(word, 'NN') for word in tokens]
            else:
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                tokens = text.split()
                tagged = [(word, 'NN') for word in tokens]
            
            # NER processing with timeout
            try:
                ner_results = self.ner_model(text, timeout=5)  # 5 second timeout
                technical_entities = [entity['word'] for entity in ner_results 
                                   if entity['entity'] != 'O']
            except Exception as e:
                logger.warning(f"NER processing failed: {e}")
                technical_entities = []
            
            # Extract technical terms
            technical_terms = [word for word, pos in tagged 
                             if pos in ['NN', 'NNP', 'NNPS']]
            
            return {
                'full_text': text,
                'sentences': sentences,
                'technical_entities': technical_entities,
                'technical_terms': technical_terms,
                'tokens': tokens
            }
        except Exception as e:
            logger.error(f"Error in engine text processing: {e}")
            return {
                'full_text': text,
                'sentences': [text],
                'technical_entities': [],
                'technical_terms': [],
                'tokens': text.split()
            }

class EngineTroubleshootingRAG:
    """Handles engine troubleshooting queries using RAG approach."""
    
    def __init__(self):
        self.embedding_model = self._load_embedding_model()
        self.qdrant_client = self._init_qdrant_client()
        self.text_processor = None
        self.openai_key = self._get_api_key()
        
        if self.is_initialized():
            try:
                self.text_processor = EngineTextProcessor(
                    pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
                )
            except Exception as e:
                logger.error(f"Error initializing text processor: {e}")
        
    def is_initialized(self) -> bool:
        """Check if critical components are initialized"""
        return all([
            self.embedding_model is not None,
            self.qdrant_client is not None,
            self.openai_key is not None
        ])
        
    def _load_embedding_model(self):
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return None

    def _init_qdrant_client(self):
        """Initialize Qdrant client using st.secrets"""
        try:
            client = QdrantClient(
                url=st.secrets["qdrant"]["url"],
                api_key=st.secrets["qdrant"]["api_key"],
                timeout=30  # Increase timeout to 30 seconds
            )
            # Test the connection
            try:
                client.get_collections()
                return client
            except Exception as e:
                logger.error(f"Qdrant connection test failed: {e}")
                return None
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            return None

    def _get_api_key(self) -> Optional[str]:
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

    def _retry_qdrant_search(self, query_vector: List[float], max_retries: int = 3) -> List[Any]:
        """Retry Qdrant search with exponential backoff"""
        for attempt in range(max_retries):
            try:
                return self.qdrant_client.search(
                    collection_name="manual_vectors",
                    query_vector=query_vector,
                    limit=10,
                    query_filter=Filter(
                        must=[
                            FieldCondition(key="type", match={"value": "engine"})
                        ]
                    ),
                    timeout=10  # 10 second timeout per attempt
                )
            except Exception as e:
                wait_time = (2 ** attempt) + 1  # Exponential backoff
                logger.warning(f"Qdrant search attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    raise

    def process_engine_query(self, 
                           question: str, 
                           vessel_name: str = None, 
                           chat_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        """Process an engine-related query with improved error handling"""
        if not self.is_initialized():
            return ("System initialization error. Please try again later.", [])

        try:
            # Process the question
            processed_question = self.text_processor.process_text(question)
            
            # Get embedding
            query_embedding = self.embedding_model.encode(question).tolist()
            
            # Search with retries
            try:
                search_results = self._retry_qdrant_search(query_embedding)
            except Exception as e:
                logger.error(f"All Qdrant search attempts failed: {e}")
                return ("Unable to search engine documentation. Please try again.", [])

            # Process results
            context = ""
            relevant_images = []
            
            for result in search_results:
                try:
                    payload = result.payload
                    if payload["type"] == "text":
                        context += f"From {payload['file_name']}, page {payload['page']}:\n"
                        context += f"{payload['content']}\n"
                    elif payload["type"] == "image":
                        relevant_images.append(payload)
                        context += f"\nTechnical diagram: {payload['image_name']}"
                        context += f" from {payload['file_name']}, page {payload['page']}:\n"
                        context += f"Diagram context: {payload['surrounding_text']}\n"
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue

            if not context:
                return ("No relevant engine documentation found for your query.", [])

            # Prepare messages
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

            # Get GPT response with timeout
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=messages,
                    temperature=0.7,
                    timeout=30  # 30 second timeout
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
