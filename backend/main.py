# backend/main.py - Complete Backend for TrueLensAI Integration
import string
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
import httpx
import json
import numpy as np
import asyncio
import logging
import re
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from pathlib import Path
import uuid

# LangChain imports
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from contextlib import asynccontextmanager

load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - TrueLensAI specific
class Config:
    # Server Configuration
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    
    # Frontend Configuration - Updated for TrueLensAI
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")  # Vite default port
    ALLOWED_ORIGINS = [
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",  # Alternative React port
        "http://localhost:4173",  # Vite preview
        "https://truelensai-hackathon.vercel.app",  # Production deployment
        "https://*.vercel.app",  # Vercel deployments
    ]
    
    # API Keys (Placeholder - Replace with actual keys)
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "hf_placeholder_api_key")
    # GPT-OSS-20B endpoint and model
    HUGGINGFACE_ENDPOINT = os.getenv("HUGGINGFACE_ENDPOINT",
        "https://router.huggingface.co/v1/chat/completions")
    HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "openai/gpt-oss-20b:fireworks-ai")
    
    EXA_AI_API_KEY = os.getenv("EXA_AI_API_KEY", "exa_placeholder_api_key")
    CLARIFAI_API_KEY = os.getenv("CLARIFAI_API_KEY", "clarifai_placeholder_api_key")
    CLARIFAI_USER_ID = os.getenv("CLARIFAI_USER_ID", "clarifai_user_id")
    CLARIFAI_APP_ID = os.getenv("CLARIFAI_APP_ID", "clarifai_app_id")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for TrueLensAI startup and shutdown"""
    # Startup
    logger.info("🎨 Starting TrueLensAI Backend Services...")
    logger.info(f"Environment: {Config.ENVIRONMENT}")
    logger.info(f"Frontend URL: {Config.FRONTEND_URL}")
    logger.info(f"HuggingFace URL: {Config.HUGGINGFACE_ENDPOINT}")
    logger.info(f"HuggingFace URL: {Config.HUGGINGFACE_MODEL}")

    # Validate critical API keys
    missing_keys = []
    if Config.HUGGINGFACE_API_KEY == "hf_placeholder_api_key":
        missing_keys.append("HUGGINGFACE_API_KEY")
    if Config.EXA_AI_API_KEY == "exa_placeholder_api_key":
        missing_keys.append("EXA_AI_API_KEY")
    if Config.CLARIFAI_API_KEY == "clarifai_placeholder_api_key":
        missing_keys.append("CLARIFAI_API_KEY")

    if missing_keys:
        logger.warning(f"⚠️  Missing API keys: {', '.join(missing_keys)}")
        logger.warning("Please update your .env file with actual API keys for full functionality")
    else:
        logger.info("✅ All API keys configured")

    logger.info("🚀 TrueLensAI Backend Ready!")
    yield
    # Shutdown
    logger.info("🛑 Shutting down TrueLensAI Backend...")
    session_manager.sessions.clear()
    session_manager.user_memories.clear()
    logger.info("✅ Cleanup complete")

# FastAPI app initialization
app = FastAPI(
    title="TrueLensAI Image Search Agent",
    description="AI-powered art search and discovery platform",
    version="1.0.0",
    docs_url="/api/docs" if Config.ENVIRONMENT == "development" else None,
    lifespan=lifespan
)


# CORS configuration for TrueLensAI
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# =============================================
# PYDANTIC MODELS - TrueLensAI Enhanced
# =============================================

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = "anonymous"
    timestamp: Optional[datetime] = None
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def set_timestamp(cls, v):
        return v or datetime.now()

class ImageSearchParams(BaseModel):
    description: str = Field(..., description="Visual description of the desired image")
    style: str = Field(..., description="Art style (realistic, cartoon, abstract, etc.)")
    
class ImageResult(BaseModel):
    image_url: str
    page_url: str
    title: str
    marketplace: Optional[str] = None
    price: Optional[str] = None
    similarity_score: float
    purchase_steps: List[str]
    metadata: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    images: Optional[List[ImageResult]] = None
    session_id: str
    requires_input: Optional[Dict[str, str]] = None  # For missing parameters
    suggestions: Optional[List[str]] = None
    processing_time: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, str]
    uptime: float

# =============================================
# SESSION MANAGEMENT - TrueLensAI Enhanced
# =============================================

class SessionManager:
    def __init__(self):
        self.user_memories = {}  # Store memories by user_id
        self.sessions = {}  # Keep sessions for temporary state
        self.cleanup_interval = 3600  # 1 hour

    def get_or_create_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        """Get or create memory for a specific user"""
        if user_id not in self.user_memories:
            self.user_memories[user_id] = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=20  # Keep last 20 exchanges for better context
            )
        return self.user_memories[user_id]

    def get_or_create_session(self, session_id: str = None) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())

        if session_id not in self.sessions:
            self.sessions[session_id] = {
                'created_at': datetime.now(),
                'user_context': {},
                'last_activity': datetime.now()
            }
        else:
            self.sessions[session_id]['last_activity'] = datetime.now()

        return session_id

    def get_user_memory(self, user_id: str) -> ConversationBufferWindowMemory:
        """Get memory for a specific user"""
        return self.user_memories.get(user_id)
    
    def cleanup_sessions(self):
        """Remove inactive sessions and old user memories"""
        cutoff = datetime.now() - timedelta(hours=1)
        inactive_sessions = [
            sid for sid, session in self.sessions.items()
            if session['last_activity'] < cutoff
        ]
        for sid in inactive_sessions:
            del self.sessions[sid]

        # Clean up old user memories (keep for 24 hours)
        memory_cutoff = datetime.now() - timedelta(hours=24)
        old_memories = [
            uid for uid, memory in self.user_memories.items()
            if not hasattr(memory, 'last_activity') or memory.last_activity < memory_cutoff
        ]
        for uid in old_memories:
            del self.user_memories[uid]

        logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions and {len(old_memories)} old memories")

session_manager = SessionManager()

# =============================================
# GPT-OSS-20B INTEGRATION - TrueLensAI Specific
# =============================================

class TrueLensGPTOSS(LLM):
    """Custom LangChain LLM wrapper for GPT-OSS-20B integration with TrueLensAI"""

    api_key: str = ""
    endpoint_url: str = ""
    model_name: str = ""
    max_tokens: int = 512
    temperature: float = 0.3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.api_key = Config.HUGGINGFACE_API_KEY
        self.endpoint_url = Config.HUGGINGFACE_ENDPOINT
        self.model_name = Config.HUGGINGFACE_MODEL
    
    @property
    def _llm_type(self) -> str:
        return "truelens_gpt_oss"
    
    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Async call to HuggingFace GPT-OSS-20B API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TrueLensAI/1.0.0"
        }
        
        # Enhanced payload for HuggingFace model
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                response = await client.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()

                # Handle OpenAI-style response format
                if isinstance(result, dict):
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"].strip()
                    elif "generated_text" in result:
                        return result["generated_text"].strip()
                    else:
                        logger.warning(f"Unexpected response format: {result}")
                        return "I apologize, but I'm having trouble processing your request right now."
                else:
                    logger.warning(f"Unexpected response type: {type(result)}")
                    return "I apologize, but I'm having trouble processing your request right now."
                
            except httpx.TimeoutException:
                logger.error("HuggingFace API timeout")
                return "I'm experiencing high load right now. Please try again in a moment."
            except httpx.HTTPStatusError as e:
                logger.error(f"HuggingFace API HTTP error: {e.response.status_code}")
                if e.response.status_code == 503:
                    return "The AI model is currently loading. Please wait a moment and try again."
                return "I'm having technical difficulties. Please try again."
            except Exception as e:
                logger.error(f"Unexpected error in HuggingFace API call: {str(e)}")
                return "I encountered an unexpected error. Please try again."
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous wrapper for async call"""
        try:
            return asyncio.run(self._acall(prompt, stop, run_manager, **kwargs))
        except RuntimeError as e:
            if "running event loop" in str(e):
                logger.error(f"Asyncio event loop conflict: {str(e)}")
                # Return a properly formatted response for LangChain agent parsing
                return """Thought: I encountered an asyncio event loop conflict which prevents me from processing this request properly.

Final Answer: I'm experiencing a technical issue with the AI service. Please try again in a moment."""
            else:
                logger.error(f"Unexpected RuntimeError in LLM call: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM synchronous call: {str(e)}")
            # Return a properly formatted error response
            return """Thought: An unexpected error occurred while processing the request.

Final Answer: I encountered an unexpected error. Please try again."""

# =============================================
# ENHANCED SEARCH IMAGE TOOL - TrueLensAI
# =============================================

class TrueLensSearchImageTool(BaseTool):
    name: str = "Search_Image"
    description: str = """Advanced art search tool for TrueLensAI platform.
    Searches multiple art marketplaces and uses AI vision to rank results by similarity.
    Input: A single optimized search query string (e.g., "realistic mountain landscape painting").
    This tool will search for artwork matching the query and return ranked results with similarity scores."""

    def __init__(self):
        super().__init__()

    
    async def search_exa_ai_enhanced(self, query) -> List[Dict]:
        """Enhanced EXA AI search with TrueLensAI optimizations"""
        logger.info("Exa AI Start")
        headers = {
            "X-API-Key": Config.EXA_AI_API_KEY,
            "Content-Type": "application/json"
        }
        logger.info(query)

        
        payload = {
            "query": query,
            "type": "neural",
            "useAutoprompt": True,
            "includeDomains": ["https://www.shutterstock.com", "https://stock.adobe.com", "https://www.gettyimages.com", "https://www.istockphoto.com", "https://depositphotos.com", "https://pixabay.com", "https://unsplash.com", "https://www.pexels.com", "https://www.freepik.com", "https://www.vecteezy.com", "https://creativemarket.com", "https://www.etsy.com", "https://www.deviantart.com", "https://www.artstation.com", "https://www.behance.net", "https://dribbble.com", "https://www.iconfinder.com", "https://thenounproject.com", "https://icons8.com", "https://www.flaticon.com", "https://www.stockvault.net", "https://burst.shopify.com", "https://stocksnap.io", "https://picjumbo.com", "https://500px.com", "https://www.flickr.com", "https://www.artsy.net", "https://www.saatchiart.com", "https://fineartamerica.com", "https://society6.com"],
            "numResults": 30,  # Get more results for better ranking
            "excludeText": ["ai"],
            "contents": {
                "text": True,
                "livecrawl": "always"
            },
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    "https://api.exa.ai/search",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                results = response.json().get("results", [])
                
                # Filter and enhance results
                enhanced_results = []
                for result in results:
                    if self._is_valid_art_result(result):
                        enhanced_result = self._enhance_result_metadata(result)
                        enhanced_results.append(enhanced_result)
                logger.info("Exa AI Finish")
                return enhanced_results[:10]  # Return top 10
                
            except Exception as e:
                logger.error(f"EXA AI search error: {str(e)}")
                return []
    
    def _is_valid_art_result(self, result: Dict) -> bool:
        """Validate if result is suitable art content"""
        url = result.get("url", "").lower()
        title = result.get("title", "").lower()
        
        # Skip results that aren't art-related
        skip_keywords = ["tutorial", "how to", "diy", "course", "lesson"]
        if any(keyword in title for keyword in skip_keywords):
            return False
        
        # Must have image
        if not result.get("image"):
            return False
            
        return True
    
    def _enhance_result_metadata(self, result: Dict) -> Dict:
        """Extract marketplace and pricing information"""
        url = result.get("url", "")
        title = result.get("title", "")
        
        # Identify marketplace
        marketplace = "Unknown"
        
        # Extract price from title/snippet (basic extraction)
        price = None
        snippet = result.get("snippet", "")
        price_patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'£(\d+(?:,\d{3})*(?:\.\d{2})?)',
            r'€(\d+(?:,\d{3})*(?:\.\d{2})?)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, f"{title} {snippet}")
            if match:
                price = match.group(0)
                break
        
        result["marketplace"] = marketplace
        result["price"] = price
        return result
    
    async def vectorize_with_clarifai_enhanced(self, user_prompt: str, image_urls: List[str]) -> List[tuple]:
        """Enhanced Clarifai CLIP comparison with better error handling"""
        logger.info("Clarifai Start")
        headers = {
            "Authorization": f"Key {Config.CLARIFAI_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare inputs for batch processing
        inputs = []
        
        # Text input (user prompt)
        inputs.append({
            "data": {
                "text": {
                    "raw": user_prompt
                }
            }
        })
        
        # Image inputs (validate URLs first)
        valid_image_urls = []
        for image_url in image_urls[:8]:  # Process up to 8 images
            if self._is_valid_image_url(image_url):
                inputs.append({
                    "data": {
                        "image": {
                            "url": image_url
                        }
                    }
                })
                valid_image_urls.append(image_url)
        
        if not valid_image_urls:
            return []
        
        payload = {
            "user_app_id": {
                "user_id": Config.CLARIFAI_USER_ID,
                "app_id": Config.CLARIFAI_APP_ID
            },
            "model_id": "multimodal-clip-embed",
            "inputs": inputs
        }
        
        async with httpx.AsyncClient(timeout=45.0) as client:
            try:
                response = await client.post(
                    "https://clarifai.com/clarifai/main/models/image-embedder-clip",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                result = response.json()
                
                outputs = result.get("outputs", [])
                if len(outputs) < 2:
                    logger.warning("Insufficient outputs from Clarifai")
                    return []
                
                # Extract text embedding
                text_output = outputs[0]
                if "embeddings" not in text_output.get("data", {}):
                    logger.warning("No text embedding found")
                    return []
                
                text_embedding = np.array(text_output["data"]["embeddings"][0]["vector"])
                similarities = []
                
                # Calculate similarities
                for i, output in enumerate(outputs[1:], 0):
                    if "embeddings" in output.get("data", {}):
                        try:
                            image_embedding = np.array(output["data"]["embeddings"][0]["vector"])
                            
                            # Cosine similarity
                            similarity = np.dot(text_embedding, image_embedding) / (
                                np.linalg.norm(text_embedding) * np.linalg.norm(image_embedding)
                            )
                            
                            # Normalize to 0-1 range and apply smoothing
                            normalized_similarity = max(0, (similarity + 1) / 2)
                            logger.info(normalized_similarity)
                            similarities.append((i, normalized_similarity))
                            
                        except Exception as e:
                            logger.warning(f"Error calculating similarity for image {i}: {str(e)}")
                            similarities.append((i, 0.0))
                logger.info("Clarifai Finish")
                return similarities
                
            except Exception as e:
                logger.error(f"Clarifai API error: {str(e)}")
                return []
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Basic validation for image URLs"""
        if not url or not isinstance(url, str):
            return False
        
        # Check if URL looks like an image
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
        url_lower = url.lower()
        
        return (
            url.startswith(('http://', 'https://')) and
            any(ext in url_lower for ext in image_extensions)
        ) or 'image' in url_lower
    
    def _generate_purchase_steps(self, marketplace: str, page_url: str) -> List[str]:
        """Generate marketplace-specific purchase instructions"""
        marketplace_lower = marketplace.lower()
        
        if "etsy" in marketplace_lower:
            return [
                "Click 'Add to cart' on the Etsy listing",
                "Review item details and shipping options",
                "Proceed to secure Etsy checkout",
                "Complete payment with your preferred method",
                "Track your order through Etsy messages"
            ]
        elif "artfinder" in marketplace_lower:
            return [
                "Click 'Buy Now' or 'Add to Basket'",
                "Create ArtFinder account or sign in",
                "Review artwork details and shipping",
                "Complete secure payment process",
                "Receive confirmation and tracking info"
            ]
        elif "saatchi" in marketplace_lower:
            return [
                "Click 'Purchase' on the Saatchi Art page",
                "Sign in or create your account",
                "Choose framing and shipping options",
                "Complete payment through secure checkout",
                "Track delivery through your account"
            ]
        else:
            return [
                f"Visit the {marketplace} listing page",
                "Review artwork details and pricing",
                "Follow the site's purchase process",
                "Complete payment through secure checkout",
                "Save confirmation details for tracking"
            ]
    
    def _run(self, query: str) -> str:
        try:
            return asyncio.run(self._arun(query))
        except RuntimeError as e:
            if "running event loop" in str(e):
                return json.dumps({
                    "status": "error",
                    "message": "Asyncio loop conflict. Please try again."
                })
            else:
                raise
    
    async def _arun(self, query: str) -> str:
        start_time = datetime.now()

        try:
            # Handle optimized query string input from model
            if isinstance(query, str) and not query.startswith('{'):
                # Direct optimized query string from model
                search_query = query.strip()
                # Extract description and style for user_prompt (basic extraction)
                requirements = extract_requirements_ai(search_query)
                description = requirements.get('description', search_query)
                style = requirements.get('style', 'artwork')
                user_prompt = f"A {style} style {description}"
            else:
                # Fallback: Parse search parameters (for backward compatibility)
                if isinstance(query, str):
                    try:
                        search_params = json.loads(query)
                    except json.JSONDecodeError:
                        return json.dumps({
                            "status": "error",
                            "message": "Invalid search parameters format"
                        })
                else:
                    search_params = query

                description = search_params.get("description", "").strip()
                style = search_params.get("style", "").strip()

                if not all([description, style]):
                    return json.dumps({
                        "status": "error",
                        "message": "Missing required parameters: description, or style"
                    })

                # Construct enhanced search query
                search_query = f"{style} {description}".strip()
                user_prompt = f"A {style} style {description}"
            
            logger.info(f"TrueLensAI search: {search_query}")
            
            # Search EXA AI with enhancements
            search_results = await self.search_exa_ai_enhanced(search_query)
            
            if not search_results:
                return json.dumps({
                    "status": "no_results",
                    "message": "No artwork found matching your criteria. Try different keywords or styles.",
                    "suggestions": [
                        "Try broader style terms (e.g., 'modern' instead of 'neo-minimalist')",
                        "Use simpler descriptions",
                        "Consider alternative purposes"
                    ]
                })
            
            # Process results and extract image data
            image_data = []
            image_urls = []
            
            for result in search_results:
                if result.get("image"):
                    image_data.append({
                        "image_url": result["image"],
                        "page_url": result.get("url", ""),
                        "title": result.get("title", "Untitled Artwork"),
                        "marketplace": result.get("marketplace", "Unknown"),
                        "price": result.get("price"),
                        "snippet": result.get("snippet", ""),
                        "similarity_score": 0.0  # Will be updated
                    })
                    image_urls.append(result["image"])
            
            if not image_urls:
                return json.dumps({
                    "status": "no_images",
                    "message": "Found artwork listings but no images to analyze"
                })
            
            # Get AI-powered similarity scores
            similarities = await self.vectorize_with_clarifai_enhanced(user_prompt, image_urls)
            
            # Combine results with similarity scores
            final_results = []
            for idx, (img_idx, similarity) in enumerate(similarities):
                if img_idx < len(image_data):
                    result = image_data[img_idx].copy()
                    result["similarity_score"] = similarity
                    result["purchase_steps"] = self._generate_purchase_steps(
                        result["marketplace"], 
                        result["page_url"]
                    )
                    
                    # Add confidence level
                    if similarity > 0.8:
                        result["confidence"] = "high"
                    elif similarity > 0.6:
                        result["confidence"] = "medium"
                    else:
                        result["confidence"] = "low"
                    
                    final_results.append(result)
            
            # Sort by similarity (highest first)
            final_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            # Processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = {
                "status": "success",
                "query": search_query,
                "results": final_results[:6],  # Top 6 results
                "total_found": len(final_results),
                "processing_time": processing_time,
                "suggestions": [
                    "Click on high-confidence results for best matches",
                    "Use the provided purchase links to buy directly",
                    "Contact sellers for custom requests or questions"
                ]
            }
            
            logger.info(f"TrueLensAI search completed: {len(final_results)} results in {processing_time:.2f}s")
            return json.dumps(response, default=str)
            
        except Exception as e:
            logger.error(f"TrueLensAI search tool error: {str(e)}")
            return json.dumps({
                "status": "error",
                "message": f"Search error: {str(e)}",
                "suggestions": ["Try a simpler search query", "Check your internet connection"]
            })

# =============================================
# TRUELENSAI SYSTEM PROMPT & AGENT
# =============================================

TRUELENS_SYSTEM_PROMPT = f""""
You are a professional Art Search Assistant named TrueLens. Your primary function is to facilitate image searches for users. Your process is iterative and conversational.

Core Directives:

Requirement Verification: You must verify that the user's request contains the following two essential components:

General Image Description: A detailed description of the subject, scene, or objects within the desired image.

General Image Style: The artistic style of the image (e.g., Cartoon, Realistic, Abstract, Minimalist, Watercolor, Oil Painting, etc.).

Iterative Understanding: You will build upon and refine your understanding of the user's request with each new response they provide. If a user provides new information that fills in a missing component, you will update your internal state accordingly.

Completion and Action:

Once the user has provided a clear General Image Description and General Image Style, you are to process their query.

Format the complete query into a single, optimized string for a neural vector search that will be used as input for Exa AI search.

Once the optimized query is formatted, you MUST call the TrueLensSearchImageTool with the optimized query string as the input parameter.

IMPORTANT: When calling the tool, use this exact format:
Action: Search_Image
Action Input: "your optimized query string here"

Handling Missing Information:

If the user's request is missing the General Image Description or the General Image Style, provide a concise, one-line suggestion to guide the user. The suggestion should highlight the missing element.

Example suggestions:

If missing the description: "Please provide a more detailed description of the image you have in mind."

If missing the style: "Could you please specify the art style you are looking for (e.g., cartoon, realistic, oil painting)?"
"""

# Initialize LLM and tools for TrueLensAI
def create_truelens_agent(user_id: str):
    """Create a TrueLensAI-specific agent instance"""
    llm = TrueLensGPTOSS()
    tools = [TrueLensSearchImageTool()]
    memory = session_manager.get_or_create_memory(user_id)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        agent_kwargs={
            "system_message": TRUELENS_SYSTEM_PROMPT
        }
    )

    return agent

# =============================================
# HELPER FUNCTIONS - TrueLensAI Enhanced
# =============================================

def extract_requirements_ai(text: str) -> Dict[str, Optional[str]]:
    """AI-powered extraction of description, style"""
    text_lower = text.lower()
    
    style_keywords = {
        'realistic': ['realistic', 'photorealistic', 'detailed', 'lifelike'],
        'abstract': ['abstract', 'non-representational', 'conceptual'],
        'minimalist': ['minimalist', 'simple', 'clean', 'minimal'],
        'modern': ['modern', 'contemporary', 'current'],
        'cartoon': ['cartoon', 'animated', 'comic', 'illustration'],
        'vintage': ['vintage', 'retro', 'classic', 'old-style']
    }
    # Extract style
    style = None
    for key, keywords in style_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            style = key
            break
    
    # Description is the full text, cleaned up
    description = text.strip()
    
    return {
        'description': description,
        'style': style
    }

def format_truelens_response(response: str, processing_time: float = None) -> ChatResponse:
    """Format response for TrueLensAI frontend"""
    try:
        # Check if response contains JSON search results
        if '"status": "success"' in response or response.startswith('{"status":'):
            # Extract JSON from response
            if not response.startswith('{'):
                json_start = response.find('{"status"')
                json_end = response.rfind('}') + 1
                
                if json_start != -1 and json_end != -1:
                    json_str = response[json_start:json_end]
                else:
                    logger.warning(f"Could not extract JSON from response: {response[:100]}...")
                    return ChatResponse(
                        response="I found some artwork but couldn't process the results properly. Please try again.",
                        images=None,
                        session_id="default",
                        processing_time=processing_time
                    )
            else:
                json_str = response
                
            try:
                search_results = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}, content: {json_str[:100]}...")
                return ChatResponse(
                    response="I found some artwork but couldn't process the results properly. Please try again.",
                    images=None,
                    session_id="default",
                    processing_time=processing_time
                )
                
            # Format images for frontend
            images = []
            for result in search_results.get("results", []):
                images.append(ImageResult(
                    image_url=result["image_url"],
                    page_url=result["page_url"],
                    title=result["title"],
                    marketplace=result.get("marketplace"),
                    price=result.get("price"),
                    similarity_score=result["similarity_score"],
                    purchase_steps=result.get("purchase_steps", []),
                    metadata={
                        "confidence": result.get("confidence", "medium"),
                        "snippet": result.get("snippet", "")
                    }
                ))
            
            # Create response text
            response_text = f"🎨 **TrueLensAI found {len(images)} perfect matches!**\n\n"
            response_text += "Using advanced AI vision technology, I've ranked these artworks by similarity to your request:\n\n"
            
            for i, img in enumerate(images[:3], 1):
                confidence_emoji = "🔥" if img.metadata.get("confidence") == "high" else "✨" if img.metadata.get("confidence") == "medium" else "💫"
                response_text += f"{confidence_emoji} **{img.title}** - {img.similarity_score*100:.0f}% match </br> \n"
                if img.price:
                    response_text += f"   💰 {img.price} on {img.marketplace}\n </br> "
            
            response_text += f"\n🛒 Click any artwork above to purchase directly from the marketplace!"
            
            return ChatResponse(
                response=response_text,
                images=images,
                session_id="default",
                processing_time=processing_time,
                suggestions=search_results.get("suggestions", [])
            )
        
        # Handle cases where user needs to provide more information
        elif any(phrase in response.lower() for phrase in ["need", "missing", "provide", "tell me"]):
            # Extract what's missing
            missing_info = {}
            if "style" in response.lower():
                missing_info["style"] = "What style do you prefer? (realistic, cartoon, abstract, etc.)"
            if "description" in response.lower() or "depict" in response.lower():
                missing_info["description"] = "What should the artwork look like?"
            
            return ChatResponse(
                response=response,
                images=None,
                session_id="default",
                requires_input=missing_info if missing_info else None,
                processing_time=processing_time,
                suggestions=[
                    "I need a realistic mountain logo for my hiking company",
                    "Looking for abstract art to decorate my modern living room",
                    "Want a cartoon cat poster for my child's bedroom"
                ]
            )
        
        # Regular conversational response
        return ChatResponse(
            response=response,
            images=None,
            session_id="default",
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error formatting TrueLensAI response: {str(e)}")
        return ChatResponse(
            response=response,
            images=None,
            session_id="default",
            processing_time=processing_time
        )

def generate_suggestions_for_context(user_message: str) -> List[str]:
    """Generate contextual suggestions based on user input"""
    message_lower = user_message.lower()
    
    if "logo" in message_lower:
        return [
            "I need a minimalist mountain logo for my outdoor company",
            "Looking for a modern geometric logo in blue and white",
            "Want a vintage-style logo with a tree symbol"
        ]
    elif any(word in message_lower for word in ["room", "wall", "decor", "home"]):
        return [
            "I want abstract art for my modern living room in earth tones",
            "Looking for realistic nature photography for my bedroom wall",
            "Need colorful geometric art for my office space"
        ]
    elif "poster" in message_lower:
        return [
            "I want a vintage travel poster of Paris in retro style",
            "Looking for a motivational poster with bold typography",
            "Need an abstract art poster in black and white"
        ]
    else:
        return [
            "I need a realistic cat portrait for my living room",
            "Looking for a cartoon-style logo of a mountain",
            "Want abstract geometric art in bright colors"
        ]

# =============================================
# API ENDPOINTS - TrueLensAI Integration
# =============================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with TrueLensAI branding"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TrueLensAI Backend</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
            .logo { font-size: 48px; color: #4F46E5; margin-bottom: 20px; }
            .subtitle { color: #6B7280; font-size: 18px; }
        </style>
    </head>
    <body>
        <div class="logo">🎨 TrueLensAI</div>
        <h1>AI-Powered Art Discovery Platform</h1>
        <p class="subtitle">Backend API is running successfully</p>
        <p><a href="/api/docs">View API Documentation</a></p>
    </body>
    </html>
    """

@app.post("/api/chat", response_model=ChatResponse)
async def truelens_chat(message: ChatMessage, background_tasks: BackgroundTasks):
    """Main chat endpoint for TrueLensAI"""
    start_time = datetime.now()

    try:
        # Get or create session
        session_id = session_manager.get_or_create_session(message.session_id)

        # Get or create memory for this user
        user_id = message.user_id or "anonymous"
        memory = session_manager.get_or_create_memory(user_id)

        # Log the interaction
        logger.info(f"TrueLensAI Chat - Session: {session_id[:8]}, User: {user_id}, Message: {message.message[:50]}...")

        # Create agent for TrueLensAI
        agent = create_truelens_agent(user_id)

        try:
            # Log the incoming message
            logger.info(f"Processing message: {message.message}")

            # Use the agent to process the message - this allows the model to format optimized query and call tool
            agent_response = await asyncio.get_event_loop().run_in_executor(
                None, agent.run, message.message
            )

            logger.info(f"Agent response: {agent_response[:200]}...")

            # Extract requirements from user message for fallback handling
            requirements = extract_requirements_ai(message.message)

            # Check if we have all requirements and agent didn't successfully call tool
            has_all_requirements = all([
                requirements.get('description'),
                requirements.get('style')
            ])

            # If we have requirements but agent failed to call tool properly, do it manually
            if has_all_requirements and not ('"status": "success"' in agent_response or agent_response.startswith('{"status":')):
                logger.info("Agent failed to call tool, falling back to manual tool call")

                # Format optimized query
                optimized_query = f"{requirements['style']} {requirements['description']}".strip()

                # Call the tool directly
                tool_response = await TrueLensSearchImageTool()._arun(optimized_query)

                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()

                # Format the tool response
                formatted_response = format_truelens_response(tool_response, processing_time)
                formatted_response.session_id = session_id

            # Check if agent response contains tool results (JSON response from tool)
            if '"status": "success"' in agent_response or agent_response.startswith('{"status":'):
                logger.info("Agent used search tool with optimized query")

                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()

                # Format the tool response
                formatted_response = format_truelens_response(agent_response, processing_time)
                formatted_response.session_id = session_id

            elif "Search_Image" in agent_response and ("Action:" in agent_response or "Action Input:" in agent_response):
                # Agent is trying to call the tool but hasn't executed it yet
                logger.info("Agent is attempting to call tool, but tool execution failed")

                # Try to extract the query from the agent response
                try:
                    # Look for the action input
                    if "Action Input:" in agent_response:
                        input_start = agent_response.find("Action Input:") + len("Action Input:")
                        query_part = agent_response[input_start:].strip()
                        # Remove quotes if present
                        if query_part.startswith('"') and query_part.endswith('"'):
                            query_part = query_part[1:-1]

                        logger.info(f"Extracted query from agent: {query_part}")

                        # Call the tool directly with the extracted query
                        tool_response = await TrueLensSearchImageTool()._arun(query_part)

                        # Calculate processing time
                        processing_time = (datetime.now() - start_time).total_seconds()

                        # Format the tool response
                        formatted_response = format_truelens_response(tool_response, processing_time)
                        formatted_response.session_id = session_id
                    else:
                        raise ValueError("Could not extract query from agent response")
                except Exception as e:
                    logger.error(f"Failed to extract and execute tool call: {str(e)}")
                    # Fall back to conversational response
                    processing_time = (datetime.now() - start_time).total_seconds()
                    formatted_response = ChatResponse(
                        response="I understand you want to search for artwork, but I'm having trouble processing your request. Could you please provide more details about what you're looking for?",
                        images=None,
                        session_id=session_id,
                        processing_time=processing_time,
                        suggestions=generate_suggestions_for_context(message.message)
                    )

            else:
                # Agent provided conversational response
                logger.info("Agent provided conversational response")

                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds()

                # Format the agent response
                formatted_response = ChatResponse(
                    response=agent_response,
                    images=None,
                    session_id=session_id,
                    processing_time=processing_time,
                    suggestions=generate_suggestions_for_context(message.message)
                )

                # Check if we need to ask for specific information
                missing_info = {}
                if not requirements.get('style'):
                    missing_info["style"] = "What style do you prefer? (realistic, cartoon, abstract, etc.)"
                if not requirements.get('description'):
                    missing_info["description"] = "What should the artwork look like?"

                if missing_info:
                    formatted_response.requires_input = missing_info
        
        except asyncio.TimeoutError:
            response = "I'm taking longer than expected to process your request. This might be due to high demand on our AI services. Please try again with a simpler request."
            formatted_response = ChatResponse(
                response=response,
                images=None,
                session_id=session_id,
                processing_time=(datetime.now() - start_time).total_seconds(),
                suggestions=generate_suggestions_for_context(message.message)
            )
        except Exception as error:
            logger.error(f"Processing error: {str(error)}")
            response = "I had trouble processing that request. Please try rephrasing your message or try again in a moment."
            formatted_response = ChatResponse(
                response=response,
                images=None,
                session_id=session_id,
                processing_time=(datetime.now() - start_time).total_seconds(),
                suggestions=generate_suggestions_for_context(message.message)
            )
        
        # Add contextual suggestions if none provided
        if not formatted_response.suggestions:
            formatted_response.suggestions = generate_suggestions_for_context(message.message)
        
        # Background cleanup
        if len(session_manager.sessions) > 100:
            background_tasks.add_task(session_manager.cleanup_sessions)
        
        return formatted_response
        
    except Exception as e:
        logger.error(f"TrueLensAI chat error: {str(e)}")
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ChatResponse(
            response=(f"I encountered an unexpected error while processing your request. Our AI services might be experiencing high load. Please try again in a moment. 🤖: {str(e)}"),
            images=None,
            session_id=session_manager.get_or_create_session(message.session_id),
            processing_time=processing_time,
            suggestions=generate_suggestions_for_context(message.message)
        )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check for TrueLensAI"""
    start_time = datetime.now()
    
    # Check service availability
    services = {
        "huggingface": "unknown",
        "exa_ai": "unknown", 
        "clarifai": "unknown",
        "sessions": "active"
    }
    
    # Quick HuggingFace check
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                Config.HUGGINGFACE_ENDPOINT,
                headers={"Authorization": f"Bearer {Config.HUGGINGFACE_API_KEY}"}
            )
            services["huggingface"] = "available" if response.status_code in [200, 503] else "error"
    except:
        services["huggingface"] = "timeout"
    
    # Session manager status
    services["sessions"] = f"active ({len(session_manager.sessions)} sessions)"
    
    # Mock checks for other services (replace with real checks)
    services["exa_ai"] = "configured" if Config.EXA_AI_API_KEY != "exa_placeholder_api_key" else "not_configured"
    services["clarifai"] = "configured" if Config.CLARIFAI_API_KEY != "clarifai_placeholder_api_key" else "not_configured"
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        services=services,
        uptime=(datetime.now() - start_time).total_seconds()
    )

@app.get("/api/users/{user_id}/history")
async def get_user_history(user_id: str):
    """Get chat history for a specific user"""
    memory = session_manager.get_user_memory(user_id)
    if not memory:
        raise HTTPException(status_code=404, detail="User memory not found")

    # Extract chat history
    messages = memory.chat_memory.messages
    history = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({"type": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"type": "assistant", "content": msg.content})

    return {"user_id": user_id, "history": history}

@app.delete("/api/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear a specific session"""
    if session_id in session_manager.sessions:
        del session_manager.sessions[session_id]
        return {"message": f"Session {session_id} cleared"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@app.get("/api/stats")
async def get_stats():
    """Get platform statistics"""
    return {
        "active_sessions": len(session_manager.sessions),
        "active_user_memories": len(session_manager.user_memories),
        "total_interactions": sum(
            len(memory.chat_memory.messages)
            for memory in session_manager.user_memories.values()
        ),
        "uptime": "N/A",  # Implement actual uptime tracking
        "version": "1.0.0",
        "environment": Config.ENVIRONMENT
    }


# =============================================
# ERROR HANDLERS
# =============================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Check the API documentation at /api/docs"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {"error": "Internal server error", "message": "Please try again later"}

# =============================================
# MAIN APPLICATION ENTRY POINT
# =============================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("🎨 Starting TrueLensAI Development Server...")
    
    uvicorn.run(
        "main:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.ENVIRONMENT == "development",
        log_level="info"
    )
