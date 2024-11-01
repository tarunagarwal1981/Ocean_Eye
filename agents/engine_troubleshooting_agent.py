import logging
from typing import List, Dict, Tuple, Any
import openai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
import os
import nltk
from transformers import pipeline
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EngineTextProcessor:
    """Processes engine-related text with specialized technical understanding."""
    
    def __init__(self, ner_model):
        self.ner_model = ner_model
        
    def process_text(self, text: str) -> Dict[str, Any]:
        try:
            # Basic text processing
            sentences = nltk.sent_tokenize(text)
            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            
            # Named entity recognition focusing on technical terms
            ner_results = self.ner_model(text)
            technical_entities = [entity['word'] for entity in ner_results 
                               if entity['entity'] != 'O']
            
            # Extract technical terms and potential component names
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
        self.text_processor = EngineTextProcessor(
            pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        )
        openai.api_key = self._get_api_key()
        
    def _load_embedding_model(self):
        try:
            return SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            return None

    def _init_qdrant_client(self):
        try:
            return QdrantClient(
                url=st.secrets["qdrant"]["url"],
                api_key=st.secrets["qdrant"]["api_key"]
            )
        except Exception as e:
            logger.error(f"Qdrant initialization failed: {e}")
            return None

    def _get_api_key(self):
        if 'openai' in st.secrets:
            return st.secrets['openai']['api_key']
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found")
        return api_key

    def process_engine_query(self, 
                           question: str, 
                           vessel_name: str = None, 
                           chat_history: List[Dict] = None) -> Tuple[str, List[Dict]]:
        """
        Process an engine-related query and return relevant answer and images.
        
        Args:
            question (str): The user's question about engine troubleshooting
            vessel_name (str, optional): Specific vessel name if applicable
            chat_history (List[Dict], optional): Previous conversation history
            
        Returns:
            Tuple[str, List[Dict]]: Answer and relevant images
        """
        try:
            # Process the question
            processed_question = self.text_processor.process_text(question)
            
            # Get embedding for the question
            query_embedding = self.embedding_model.encode(question).tolist()
            
            # Search in Qdrant with engine-specific filter
            search_results = self.qdrant_client.search(
                collection_name="manual_vectors",
                query_vector=query_embedding,
                limit=10,
                query_filter=Filter(
                    must=[
                        FieldCondition(key="type", match={"value": "engine"})
                    ]
                )
            )

            # Process results
            context = ""
            relevant_images = []
            
            for result in search_results:
                payload = result.payload
                if payload["type"] == "text":
                    context += f"From {payload['file_name']}, page {payload['page']}:\n"
                    context += f"{payload['content']}\n"
                elif payload["type"] == "image":
                    relevant_images.append(payload)
                    context += f"\nTechnical diagram: {payload['image_name']}"
                    context += f" from {payload['file_name']}, page {payload['page']}:\n"
                    context += f"Diagram context: {payload['surrounding_text']}\n"

            # Prepare chat messages with engine expertise
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
                
            # Add vessel context if provided
            vessel_context = f" for vessel {vessel_name}" if vessel_name else ""
            
            messages.extend([
                {"role": "user", "content": 
                 f"Context:\n{context}\n\nQuestion about engine troubleshooting{vessel_context}: {question}"}
            ])

            # Get response from GPT-4
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7
            )
            
            answer = response.choices[0].message['content'].strip()
            return answer, relevant_images

        except Exception as e:
            logger.error(f"Error processing engine query: {e}")
            return "Sorry, there was an error processing your engine troubleshooting query.", []

def analyze_engine_troubleshooting(vessel_name: str, query: str) -> Tuple[str, List[Dict]]:
    """
    Main function to analyze engine troubleshooting queries.
    
    Args:
        vessel_name (str): Name of the vessel
        query (str): User's engine-related query
        
    Returns:
        Tuple[str, List[Dict]]: Answer and relevant technical diagrams
    """
    try:
        rag = EngineTroubleshootingRAG()
        return rag.process_engine_query(query, vessel_name)
    except Exception as e:
        logger.error(f"Error in engine troubleshooting analysis: {e}")
        return "Error processing engine troubleshooting query.", []
