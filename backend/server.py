from fastapi import FastAPI, APIRouter, HTTPException, Response, Cookie, Header, Request
from fastapi.responses import JSONResponse, RedirectResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
from openai import AsyncOpenAI
import aiohttp
from authlib.integrations.starlette_client import OAuth
import asyncio

# Optional local NLP trainer module (depends on sentence-transformers + scikit-learn)
try:
    from nlp import trainer
except Exception:
    trainer = None

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MongoDB connection (use defaults and avoid crashing on missing env vars)
# If MONGO_URL/DB_NAME are not provided, fall back to localhost defaults so
# the server can start in development. We attempt a ping on startup and log
# any connectivity issues without preventing the app from running.
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
db_name = os.environ.get('DB_NAME', 'ai_interview')
client = None
db = None
try:
    # Set a short serverSelectionTimeout to fail fast if Mongo isn't reachable
    client = AsyncIOMotorClient(mongo_url, serverSelectionTimeoutMS=5000)
    db = client[db_name]
except Exception as e:
    logger.warning(f"Failed to initialize MongoDB client: {e}")
    client = None
    db = None

# Create the main app
app = FastAPI()
api_router = APIRouter(prefix="/api")

# OpenAI API Key - can use EMERGENT_LLM_KEY or OPENAI_API_KEY
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

openai_client = None
use_emergent_api = False

if OPENAI_API_KEY:
    # Prefer standard OpenAI API
    try:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("Using OpenAI API")
        use_emergent_api = False
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
elif EMERGENT_LLM_KEY:
    # Fallback to Emergent Agent API endpoint
    try:
        openai_client = AsyncOpenAI(
            api_key=EMERGENT_LLM_KEY,
            base_url="https://demobackend.emergentagent.com/llm/v1"
        )
        logger.info("Using Emergent LLM API")
        use_emergent_api = True
    except Exception as e:
        logger.error(f"Failed to initialize Emergent client: {e}")

if not openai_client:
    logger.warning("No AI client configured. Set OPENAI_API_KEY or EMERGENT_LLM_KEY environment variable.")

# Google OAuth Configuration
oauth = OAuth()

# Only register OAuth if credentials are provided
if os.environ.get('GOOGLE_CLIENT_ID') and os.environ.get('GOOGLE_CLIENT_SECRET'):
    oauth.register(
        name='google',
        client_id=os.environ.get('GOOGLE_CLIENT_ID'),
        client_secret=os.environ.get('GOOGLE_CLIENT_SECRET'),
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid email profile'
        }
    )

# Models
class User(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    picture: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Session(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    session_token: str
    expires_at: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class InterviewSession(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    interview_type: str  # "technical", "behavioral", "general"
    status: str = "active"  # "active", "completed", "evaluated"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interview_id: str
    role: str  # "user" or "assistant"
    content: str
    is_voice: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Evaluation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    interview_id: str
    overall_score: float
    communication_score: float
    technical_score: float
    problem_solving_score: float
    strengths: List[str]
    areas_for_improvement: List[str]
    detailed_feedback: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# Request/Response Models
class SessionRequest(BaseModel):
    session_id: str

class CreateInterviewRequest(BaseModel):
    interview_type: str

class SendMessageRequest(BaseModel):
    content: str
    is_voice: bool = False

class VoiceTranscribeRequest(BaseModel):
    audio_text: str

# Helper function to get current user from session
async def get_current_user(session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    token = session_token
    if not token and authorization:
        if authorization.startswith('Bearer '):
            token = authorization[7:]
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = await db.sessions.find_one({"session_token": token})
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    
    # Check if session is expired
    expires_at = session.get('expires_at')
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    
    user = await db.users.find_one({"id": session['user_id']}, {"_id": 0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return User(**user)

# Auth endpoints
@api_router.post("/auth/session")
async def create_session(request: SessionRequest):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            'https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data',
            headers={'X-Session-ID': request.session_id}
        ) as resp:
            if resp.status != 200:
                raise HTTPException(status_code=400, detail="Invalid session ID")
            data = await resp.json()
    
    # Check if user exists
    existing_user = await db.users.find_one({"email": data['email']}, {"_id": 0})
    if not existing_user:
        user = User(
            id=str(uuid.uuid4()),
            email=data['email'],
            name=data['name'],
            picture=data.get('picture')
        )
        user_dict = user.model_dump()
        user_dict['created_at'] = user_dict['created_at'].isoformat()
        await db.users.insert_one(user_dict)
    else:
        user = User(**existing_user)
    
    # Create session
    session_token = data['session_token']
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    session_obj = Session(
        id=str(uuid.uuid4()),
        user_id=user.id,
        session_token=session_token,
        expires_at=expires_at
    )
    
    session_dict = session_obj.model_dump()
    session_dict['expires_at'] = session_dict['expires_at'].isoformat()
    session_dict['created_at'] = session_dict['created_at'].isoformat()
    await db.sessions.insert_one(session_dict)
    
    response = JSONResponse(content={
        "user": user.model_dump(mode='json'),
        "session_token": session_token
    })
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=True,
        samesite="none",
        path="/",
        max_age=7*24*60*60
    )
    return response

# Development-only endpoint for testing without OAuth
@api_router.post("/auth/dev-login")
async def dev_login():
    """Create a test user session for development purposes"""
    test_email = "test@example.com"
    
    # Check if test user exists
    existing_user = await db.users.find_one({"email": test_email}, {"_id": 0})
    if not existing_user:
        user = User(
            id=str(uuid.uuid4()),
            email=test_email,
            name="Test User",
            picture="https://ui-avatars.com/api/?name=Test+User"
        )
        user_dict = user.model_dump()
        user_dict['created_at'] = user_dict['created_at'].isoformat()
        await db.users.insert_one(user_dict)
    else:
        user = User(**existing_user)
    
    # Create session
    session_token = str(uuid.uuid4())
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    session_obj = Session(
        id=str(uuid.uuid4()),
        user_id=user.id,
        session_token=session_token,
        expires_at=expires_at
    )
    
    session_dict = session_obj.model_dump()
    session_dict['expires_at'] = session_dict['expires_at'].isoformat()
    session_dict['created_at'] = session_dict['created_at'].isoformat()
    await db.sessions.insert_one(session_dict)
    
    response = JSONResponse(content={
        "user": user.model_dump(mode='json'),
        "session_token": session_token
    })
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=False,  # Allow for localhost
        samesite="lax",
        path="/",
        max_age=7*24*60*60
    )
    return response

# Google OAuth endpoints
@api_router.get("/auth/google/login")
async def google_login(request: Request):
    """Redirect to Google OAuth login page"""
    redirect_uri = os.environ.get('GOOGLE_REDIRECT_URI', 'http://localhost:8000/api/auth/google/callback')
    return await oauth.google.authorize_redirect(request, redirect_uri)

@api_router.get("/auth/google/callback")
async def google_callback(request: Request):
    """Handle Google OAuth callback"""
    try:
        # Get the token from Google
        token = await oauth.google.authorize_access_token(request)
        
        # Get user info from Google
        user_info = token.get('userinfo')
        if not user_info:
            raise HTTPException(status_code=400, detail="Failed to get user info from Google")
        
        email = user_info.get('email')
        name = user_info.get('name')
        picture = user_info.get('picture')
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by Google")
        
        # Check if user exists
        existing_user = await db.users.find_one({"email": email}, {"_id": 0})
        if not existing_user:
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                name=name or email.split('@')[0],
                picture=picture
            )
            user_dict = user.model_dump()
            user_dict['created_at'] = user_dict['created_at'].isoformat()
            await db.users.insert_one(user_dict)
        else:
            user = User(**existing_user)
            # Update picture if it changed
            if picture and existing_user.get('picture') != picture:
                await db.users.update_one(
                    {"email": email},
                    {"$set": {"picture": picture}}
                )
                user.picture = picture
        
        # Create session
        session_token = str(uuid.uuid4())
        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        session_obj = Session(
            id=str(uuid.uuid4()),
            user_id=user.id,
            session_token=session_token,
            expires_at=expires_at
        )
        
        session_dict = session_obj.model_dump()
        session_dict['expires_at'] = session_dict['expires_at'].isoformat()
        session_dict['created_at'] = session_dict['created_at'].isoformat()
        await db.sessions.insert_one(session_dict)
        
        # Redirect to frontend with session token
        frontend_url = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
        response = RedirectResponse(url=f"{frontend_url}/auth/callback?token={session_token}")
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=False,  # Set to True in production with HTTPS
            samesite="lax",
            path="/",
            max_age=7*24*60*60
        )
        return response
        
    except Exception as e:
        logging.error(f"Google OAuth error: {str(e)}")
        frontend_url = os.environ.get('FRONTEND_URL', 'http://localhost:3000')
        return RedirectResponse(url=f"{frontend_url}/?error=auth_failed")

@api_router.get("/auth/me")
async def get_me(session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    """Get current user session. Returns user data if authenticated, null if not."""
    try:
        user = await get_current_user(session_token, authorization)
        return user
    except HTTPException as e:
        if e.status_code == 401:
            # Return null for unauthenticated users instead of 401 error
            return None
        raise

@api_router.post("/auth/logout")
async def logout(session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    token = session_token or (authorization[7:] if authorization and authorization.startswith('Bearer ') else None)
    if token:
        await db.sessions.delete_one({"session_token": token})
    
    response = JSONResponse(content={"message": "Logged out successfully"})
    response.delete_cookie(key="session_token", path="/")
    return response

# Interview endpoints
@api_router.post("/interviews")
async def create_interview(request: CreateInterviewRequest, session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    user = await get_current_user(session_token, authorization)
    
    interview = InterviewSession(
        id=str(uuid.uuid4()),
        user_id=user.id,
        interview_type=request.interview_type,
        status="active"
    )
    
    interview_dict = interview.model_dump()
    interview_dict['created_at'] = interview_dict['created_at'].isoformat()
    await db.interview_sessions.insert_one(interview_dict)
    
    # Send initial AI message
    initial_prompts = {
        "technical": "Hello! I'm excited to chat with you today about your technical background and skills. I'll be asking questions about programming, algorithms, system design, and your problem-solving approach. To get us started, could you tell me about a recent technical project you're proud of and what technologies you used?",
        "behavioral": "Hi there! Thanks for taking the time to interview today. I'm interested in learning about your work experiences and how you approach various workplace situations. Let's begin with this: Can you tell me about a time when you had to overcome a significant challenge at work? What was the situation and how did you handle it?",
        "general": "Welcome! I'm glad we have this opportunity to get to know each other. I'll be asking you various questions to understand your background, skills, and what you're looking for in your career. Let's start with an introduction - could you tell me about your professional journey so far and what brings you here today?"
    }
    
    initial_message = Message(
        id=str(uuid.uuid4()),
        interview_id=interview.id,
        role="assistant",
        content=initial_prompts.get(request.interview_type, initial_prompts["general"]),
        is_voice=False
    )
    
    msg_dict = initial_message.model_dump()
    msg_dict['timestamp'] = msg_dict['timestamp'].isoformat()
    await db.messages.insert_one(msg_dict)
    
    # Return interview with ISO formatted datetime
    interview_response = interview.model_dump()
    interview_response['created_at'] = interview_response['created_at'].isoformat()
    if interview_response.get('completed_at'):
        interview_response['completed_at'] = interview_response['completed_at'].isoformat()
    
    return interview_response

@api_router.get("/interviews")
async def get_interviews(session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    user = await get_current_user(session_token, authorization)
    
    interviews = await db.interview_sessions.find(
        {"user_id": user.id},
        {"_id": 0}
    ).sort("created_at", -1).to_list(100)
    
    # Ensure datetime fields are in ISO format for JSON serialization
    for interview in interviews:
        if isinstance(interview.get('created_at'), datetime):
            interview['created_at'] = interview['created_at'].isoformat()
        if interview.get('completed_at') and isinstance(interview['completed_at'], datetime):
            interview['completed_at'] = interview['completed_at'].isoformat()
    
    return interviews

@api_router.get("/interviews/{interview_id}")
async def get_interview(interview_id: str, session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    user = await get_current_user(session_token, authorization)
    
    interview = await db.interview_sessions.find_one(
        {"id": interview_id, "user_id": user.id},
        {"_id": 0}
    )
    
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    messages = await db.messages.find(
        {"interview_id": interview_id},
        {"_id": 0}
    ).sort("timestamp", 1).to_list(1000)
    
    # Ensure timestamps are in ISO format for JSON serialization
    for msg in messages:
        if isinstance(msg.get('timestamp'), datetime):
            msg['timestamp'] = msg['timestamp'].isoformat()
    
    if isinstance(interview.get('created_at'), datetime):
        interview['created_at'] = interview['created_at'].isoformat()
    if interview.get('completed_at') and isinstance(interview['completed_at'], datetime):
        interview['completed_at'] = interview['completed_at'].isoformat()
    
    return {
        "interview": interview,
        "messages": messages
    }

@api_router.post("/interviews/{interview_id}/message")
async def send_message(interview_id: str, request: SendMessageRequest, session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    try:
        user = await get_current_user(session_token, authorization)
        
        interview = await db.interview_sessions.find_one(
            {"id": interview_id, "user_id": user.id},
            {"_id": 0}
        )
        
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        # Check if interview is completed or evaluated
        if interview.get('status') in ['completed', 'evaluated']:
            raise HTTPException(status_code=400, detail="Cannot send messages to a completed interview")
        
        # Save user message
        user_message = Message(
            id=str(uuid.uuid4()),
            interview_id=interview_id,
            role="user",
            content=request.content,
            is_voice=request.is_voice
        )
        
        user_msg_dict = user_message.model_dump()
        user_msg_dict['timestamp'] = user_msg_dict['timestamp'].isoformat()
        await db.messages.insert_one(user_msg_dict)
        
        # Build context for AI
        interview_type = interview['interview_type']
        system_messages = {
            "technical": """You are an experienced technical interviewer conducting a real technical interview. 
Your role is to:
- Ask relevant technical questions about programming, algorithms, data structures, system design, and problem-solving
- Analyze the candidate's responses critically and provide thoughtful follow-up questions
- Vary your responses naturally - sometimes acknowledge good points, sometimes probe deeper, sometimes challenge assumptions
- Be professional, conversational, and authentic like a real interviewer
- Adapt your tone based on the candidate's answer quality (encouraging for good answers, probing for incomplete ones)
- Use varied response patterns: acknowledgments, follow-ups, new questions, requests for clarification, or examples
- Keep responses concise and natural (2-4 sentences typically)

Response variation examples:
- "Interesting approach. Have you considered [alternative]?"
- "That makes sense. Can you walk me through how you'd handle [specific scenario]?"
- "I see. What trade-offs did you consider?"
- "Good point. Let me ask you about [related topic]..."
- "Can you elaborate on [specific part of their answer]?"
""",
            "behavioral": """You are an experienced HR interviewer conducting a behavioral interview.
Your role is to:
- Ask questions about past experiences, teamwork, conflict resolution, leadership, and challenges
- Listen carefully to responses and ask natural follow-up questions
- Use the STAR method (Situation, Task, Action, Result) to guide questioning when appropriate
- Vary your responses authentically - sometimes empathize, sometimes dig deeper, sometimes move to new topics
- Be conversational and create a comfortable interview atmosphere
- Adapt your tone: supportive for personal stories, probing for details, appreciative of insights

Response variation examples:
- "That sounds challenging. How did the team respond to your decision?"
- "I appreciate you sharing that. What did you learn from that experience?"
- "Interesting. Can you tell me more about your specific role in that situation?"
- "That's a great example. Let's explore another area - [new topic]..."
- "How did you measure the success of that approach?"
""",
            "general": """You are a professional interviewer conducting a comprehensive interview.
Your role is to:
- Ask a mix of questions about background, skills, experience, and career goals
- Create a natural, flowing conversation that feels authentic
- Vary your responses based on what the candidate shares
- Be genuinely interested and conversational
- Transition smoothly between topics
- Mix acknowledgments, follow-ups, and new questions naturally

Response variation examples:
- "That's really interesting. What motivated you to [specific choice they mentioned]?"
- "I see. How does that align with your long-term career goals?"
- "Thanks for sharing that background. Let me ask you about [different topic]..."
- "Great. Can you give me an example of when you used [skill they mentioned]?"
- "That makes sense. What would you say is your biggest strength in [area]?"
"""
        }
        
        # Get conversation history
        messages_cursor = db.messages.find({"interview_id": interview_id}).sort("timestamp", 1)
        conversation_history = []
        async for msg in messages_cursor:
            conversation_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Analyze user's message to generate context-aware response
        user_content = request.content.lower()
        message_length = len(request.content.split())
        
        # Detect answer characteristics for intelligent response generation
        answer_characteristics = {
            "is_detailed": message_length > 30,
            "is_brief": message_length < 10,
            "mentions_example": any(keyword in user_content for keyword in ["example", "instance", "time when", "once", "project"]),
            "mentions_technology": any(tech in user_content for tech in ["python", "javascript", "java", "react", "node", "sql", "api", "database", "algorithm", "framework", "library"]),
            "mentions_team": any(keyword in user_content for keyword in ["team", "colleague", "coworker", "manager", "collaborate"]),
            "mentions_challenge": any(keyword in user_content for keyword in ["challenge", "difficult", "problem", "issue", "struggle", "obstacle"]),
            "mentions_success": any(keyword in user_content for keyword in ["success", "achieved", "accomplished", "improved", "solved"]),
            "shows_uncertainty": any(phrase in user_content for phrase in ["i think", "maybe", "not sure", "probably", "i guess"]),
            "asks_clarification": "?" in request.content
        }
        
        # First, try NLP model if enabled and available (for learned responses)
        ai_response = None
        if os.environ.get("USE_NLP_RESPONSES") == "1" and trainer is not None:
            try:
                logger.info("Attempting to use NLP model for response")
                nlp_result = trainer.get_best_response(request.content, confidence_threshold=0.6)
                if "error" not in nlp_result or nlp_result.get("warning"):
                    confidence = nlp_result.get("confidence", 0)
                    if confidence >= 0.6:
                        ai_response = nlp_result["response"]
                        logger.info(f"Using NLP response with {confidence:.2%} confidence")
                    else:
                        logger.info(f"NLP confidence ({confidence:.2%}) below threshold, trying other methods")
                else:
                    logger.info("NLP model returned error, trying other methods")
            except Exception as e:
                logger.warning(f"Failed to get NLP response: {e}")
        
        # If NLP didn't provide a response, check if OpenAI client is available
        if not ai_response and not openai_client:
            logger.error("OpenAI client not configured")
            # Use fallback mock response for development/testing
            logger.warning("Using intelligent mock AI response - configure OPENAI_API_KEY or EMERGENT_LLM_KEY for real AI")
            
            # Generate intelligent context-aware mock responses with variety
            import random
            
            def generate_intelligent_response(interview_type, characteristics, user_msg, conv_count):
                """Generate responses that actually react to user's message content with variety"""
                
                # Extract specific words from user message
                words = user_msg.split()
                key_words = [w for w in words if len(w) > 4][:3]
                
                if interview_type == "technical":
                    responses = []
                    if characteristics["is_brief"]:
                        responses = [
                            "I need more detail - what was your specific implementation?",
                            "Could you walk me through the technical approach you took?",
                            "Expand on that. What technologies did you use and why?"
                        ]
                    elif characteristics["shows_uncertainty"]:
                        responses = [
                            "Let's break this down step by step. Where would you start?",
                            "Think through the problem architecture. What components do you need?",
                            "No wrong answers here - what's your initial approach?"
                        ]
                    elif characteristics["mentions_technology"]:
                        tech_word = key_words[0] if key_words else "that technology"
                        responses = [
                            f"What made {tech_word} the right choice over alternatives?",
                            f"How did you handle scalability with {tech_word}?",
                            f"What challenges did {tech_word} introduce to your stack?"
                        ]
                    elif characteristics["mentions_example"]:
                        responses = [
                            "Walk me through the architecture decisions you made there.",
                            "What was the trickiest part of that implementation?",
                            "How did you test and validate that solution?"
                        ]
                    elif characteristics["is_detailed"]:
                        responses = [
                            "How would this perform under 100x load?",
                            "What monitoring did you put in place?",
                            "What would you refactor if you rebuilt this today?"
                        ]
                    else:
                        responses = [
                            "Describe the data flow in that system.",
                            f"What assumptions did you make about {key_words[0] if key_words else 'the requirements'}?",
                            "How did you ensure code quality and maintainability?"
                        ]
                    return random.choice(responses)
                
                elif interview_type == "behavioral":
                    responses = []
                    if characteristics["is_brief"]:
                        responses = [
                            "Walk me through the full story - situation, actions, and outcome.",
                            "I need specifics. What exactly did you do in that scenario?",
                            "Give me more context about the situation and your role."
                        ]
                    elif characteristics["mentions_team"]:
                        responses = [
                            "How did you handle conflicts or disagreements in the team?",
                            "What was your strategy for keeping everyone aligned?",
                            "Tell me about a time when the team dynamic was challenging."
                        ]
                    elif characteristics["mentions_challenge"]:
                        responses = [
                            "What did that experience teach you professionally?",
                            "How has that shaped how you approach similar situations now?",
                            "Looking back, what would you have done differently?"
                        ]
                    elif characteristics["mentions_success"]:
                        responses = [
                            "How did you measure that success quantitatively?",
                            "What was the business impact of that achievement?",
                            "What made you successful where others might have failed?"
                        ]
                    elif characteristics["mentions_example"]:
                        responses = [
                            "What would you change if you faced that situation again?",
                            "How do you apply those learnings in your current work?",
                            "What was the lasting impact of that experience?"
                        ]
                    elif characteristics["is_detailed"]:
                        responses = [
                            "What patterns do you see across your experiences?",
                            "How do you prioritize when faced with competing demands?",
                            f"How does your approach to {key_words[0] if key_words else 'this'} differ from your peers?"
                        ]
                    else:
                        responses = [
                            "Give me a concrete example that illustrates that.",
                            "Describe a time when that skill was critical to success.",
                            "How have you developed this strength over time?"
                        ]
                    return random.choice(responses)
                
                else:  # general
                    responses = []
                    if characteristics["is_brief"]:
                        responses = [
                            "Tell me more - what specifically interests you about that?",
                            "Help me understand what drives your passion for this.",
                            "What aspect of that is most meaningful to you?"
                        ]
                    elif characteristics["asks_clarification"]:
                        responses = [
                            "Good question! I'm trying to understand what motivates your career choices.",
                            "Let me clarify - I want to know about your professional journey and goals.",
                            "To rephrase: what experiences have shaped your career direction?"
                        ]
                    elif characteristics["mentions_technology"] or characteristics["mentions_example"]:
                        responses = [
                            f"How does {key_words[0] if key_words else 'that experience'} connect to your future goals?",
                            "What skills from that are you most excited to develop further?",
                            "Where do you want to take that expertise next?"
                        ]
                    elif characteristics["is_detailed"]:
                        responses = [
                            "What are you most passionate about in your field?",
                            "How do you stay current with industry developments?",
                            "What type of impact do you want to make in your career?"
                        ]
                    else:
                        responses = [
                            "Where do you see yourself in 3-5 years?",
                            "What kind of challenges excite you most?",
                            f"How does {key_words[0] if key_words else 'this'} fit into your career vision?"
                        ]
                    return random.choice(responses)
            
            ai_response = generate_intelligent_response(interview_type, answer_characteristics, request.content, conversation_count)
            logger.info(f"Generated intelligent mock response based on answer characteristics")
        elif not ai_response and openai_client:
            # Generate AI response using OpenAI with content-aware context
            conversation_count = len([msg for msg in conversation_history if msg["role"] == "user"])
            
            # Extract previous assistant responses to avoid repetition
            previous_responses = [msg["content"] for msg in conversation_history if msg["role"] == "assistant"]
            previous_response_text = " | ".join(previous_responses[-3:]) if previous_responses else ""
            
            # Build intelligent context based on answer analysis
            context_hints = []
            if answer_characteristics["is_brief"]:
                context_hints.append("The candidate gave a brief answer - encourage them to elaborate with specific details")
            if answer_characteristics["is_detailed"]:
                context_hints.append("The candidate provided a detailed response - acknowledge depth and probe specific technical/situational aspects")
            if answer_characteristics["mentions_example"]:
                context_hints.append("They provided an example - dig into the specifics, decision-making process, and outcomes")
            if answer_characteristics["mentions_technology"]:
                context_hints.append("They mentioned specific technologies - explore technical choices, trade-offs, and alternatives")
            if answer_characteristics["mentions_team"]:
                context_hints.append("They discussed teamwork - explore collaboration dynamics, communication strategies, and conflict resolution")
            if answer_characteristics["mentions_challenge"]:
                context_hints.append("They described a challenge - explore problem-solving methodology, obstacles faced, and lessons learned")
            if answer_characteristics["mentions_success"]:
                context_hints.append("They mentioned success - probe for metrics, impact measurement, and key contributing factors")
            if answer_characteristics["shows_uncertainty"]:
                context_hints.append("They seem uncertain - guide them with clarifying questions or provide a framework to structure their thoughts")
            
            context_instruction = "\n- ".join(context_hints) if context_hints else "Respond naturally and authentically to their answer"
            
            # Extract specific nouns and technical terms from user's response
            import re
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[a-z]{4,}\b', request.content)
            key_phrases = [w for w in words if len(w) > 4][:5]  # Get up to 5 key phrases
            
            # Determine interview stage for appropriate tone
            if conversation_count <= 2:
                stage = "opening (explore background and warm up)"
            elif conversation_count <= 5:
                stage = "exploration (dig into specific experiences and skills)"
            elif conversation_count <= 8:
                stage = "deep-dive (probe technical/behavioral details and edge cases)"
            else:
                stage = "closing (synthesize insights and explore final key areas)"
            
            messages_for_api = [
                {"role": "system", "content": system_messages.get(interview_type, system_messages["general"])},
                {"role": "system", "content": f"""Current interview context (Question #{conversation_count}, Stage: {stage}):

CRITICAL ANTI-REPETITION RULES:
- Your recent responses were: "{previous_response_text}"
- You must NOT repeat any similar phrases, questions, or patterns from above
- NEVER use generic phrases like "Tell me more", "That's interesting", "I see" repeatedly
- Each response must be COMPLETELY DIFFERENT in structure and wording from previous ones

CONTENT-SPECIFIC REQUIREMENTS:
- The candidate just said: "{request.content[:200]}..."
- Key concepts they mentioned: {', '.join(key_phrases) if key_phrases else 'general discussion'}
- {context_instruction}

RESPONSE GENERATION STRATEGY:
1. REFERENCE SPECIFICS: Pull exact phrases/technologies/situations from their answer
2. VARY STRUCTURE: Rotate between these patterns (never use same twice in a row):
   - Direct technical question: "How would you handle [specific scenario from their answer]?"
   - Comparative: "What made you choose [X] over [Y alternative]?"
   - Exploratory: "Walk me through your process for [specific thing they mentioned]"
   - Challenging: "What if [edge case/complication]? How would that change your approach?"
   - Reflective: "You mentioned [specific detail] - what impact did that have on [outcome]?"
   - Probing: "Can you break down [specific technical aspect] in more detail?"
   - Scenario-based: "Imagine [related scenario] - how would you apply this experience?"
   
3. ADAPT TO STAGE: {stage} means your tone should reflect that interview phase
4. BE SPECIFIC: Replace all generic words with exact references to what they said
5. SOUND HUMAN: Use natural language, contractions, conversational flow
6. LENGTH: 2-3 sentences maximum, concise and focused

EXAMPLE TRANSFORMATIONS (what NOT to do → what TO do):
❌ "Tell me more about that" → ✅ "You mentioned using microservices - what challenges did you face with inter-service communication?"
❌ "That's interesting" → ✅ "Splitting the monolith must have been risky - how did you manage the migration without downtime?"
❌ "I see" → ✅ "So your team disagreed on the caching strategy - what data did you use to make the final call?"

Your response must be UNIQUE, SPECIFIC to their answer, and NEVER repeat patterns from your previous responses."""}
            ]
            messages_for_api.extend(conversation_history)
            
            # Try different models in order of preference
            # Use different model names depending on which API we're using
            if use_emergent_api:
                # Emergent API model names
                models_to_try = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
            else:
                # Standard OpenAI model names
                models_to_try = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"]
            
            ai_response = None
            last_error = None
            
            for model in models_to_try:
                try:
                    logger.info(f"Attempting to use model: {model}")
                    response = await openai_client.chat.completions.create(
                        model=model,
                        messages=messages_for_api,
                        temperature=1.0,  # Maximum creativity for variation
                        max_tokens=500,
                        presence_penalty=1.2,  # Very high to prevent topic repetition
                        frequency_penalty=0.8,  # High to prevent word repetition
                        top_p=0.92  # Nucleus sampling for diversity
                    )
                    
                    ai_response = response.choices[0].message.content
                    logger.info(f"Successfully generated unique contextual response using {model}")
                    break  # Success, exit the loop
                except Exception as e:
                    logger.warning(f"Failed with model {model}: {str(e)}")
                    last_error = e
                    continue  # Try next model
            
            if not ai_response:
                logger.error(f"All models failed. Last error: {str(last_error)}")
                # Fallback to intelligent mock response instead of failing
                logger.warning("All AI models failed, using intelligent context-aware fallback")
                
                # Use intelligent response generator with variety
                import random
                
                def generate_intelligent_fallback(interview_type, characteristics, user_msg, conv_count):
                    """Generate content-aware fallback responses with variation"""
                    
                    # Extract specific words from user message
                    words = user_msg.split()
                    key_words = [w for w in words if len(w) > 4][:3]
                    
                    if interview_type == "technical":
                        responses = []
                        if characteristics["is_brief"]:
                            responses = [
                                "Could you dive deeper into that? What specific approach did you take?",
                                "I need more context - walk me through the technical implementation.",
                                "Expand on that. What were the key technical considerations?"
                            ]
                        elif characteristics["mentions_technology"]:
                            tech_word = key_words[0] if key_words else "that"
                            responses = [
                                f"Why {tech_word}? What alternatives did you evaluate?",
                                f"What challenges did you face working with {tech_word}?",
                                f"How does {tech_word} fit into your overall architecture?"
                            ]
                        elif characteristics["mentions_challenge"]:
                            responses = [
                                "What was your debugging process? How did you isolate the issue?",
                                "Walk me through the root cause analysis you performed.",
                                "What would you architect differently knowing what you know now?"
                            ]
                        elif characteristics["is_detailed"]:
                            responses = [
                                "How would you scale this to handle 10x traffic?",
                                "What metrics did you use to measure performance improvements?",
                                "What edge cases did you need to handle?"
                            ]
                        else:
                            responses = [
                                "Describe your testing strategy for this.",
                                "What trade-offs did you make in your design?",
                                f"How did you validate that {key_words[0] if key_words else 'your approach'} was the right choice?"
                            ]
                        return random.choice(responses)
                    
                    elif interview_type == "behavioral":
                        responses = []
                        if characteristics["is_brief"]:
                            responses = [
                                "Give me the full STAR: situation, task, action, result.",
                                "What specifically did you do? I want to understand your individual contribution.",
                                "Set the scene for me - what was the context and what happened?"
                            ]
                        elif characteristics["mentions_success"]:
                            responses = [
                                "How did you quantify that success? What were the key metrics?",
                                "Who else contributed to this win, and what was your specific role?",
                                "What obstacles did you overcome to achieve that result?"
                            ]
                        elif characteristics["mentions_team"]:
                            responses = [
                                "How did you navigate different opinions within the team?",
                                "What was your communication strategy with stakeholders?",
                                "Tell me about a conflict that arose - how did you resolve it?"
                            ]
                        elif characteristics["mentions_challenge"]:
                            responses = [
                                "What did this teach you about yourself as a professional?",
                                "How do you apply those lessons in your current role?",
                                "What would you do differently if you could go back?"
                            ]
                        else:
                            responses = [
                                "Give me another example that shows this strength.",
                                f"How does your experience with {key_words[0] if key_words else 'this'} prepare you for larger challenges?",
                                "What's the most difficult aspect of this that you've had to master?"
                            ]
                        return random.choice(responses)
                    
                    else:  # general
                        responses = []
                        if characteristics["is_brief"]:
                            responses = [
                                "Paint me a fuller picture - what drives your interest in this?",
                                "Elaborate on that. What makes it significant to you?",
                                "Help me understand the 'why' behind that choice."
                            ]
                        elif characteristics["mentions_example"]:
                            responses = [
                                f"What did that experience with {key_words[0] if key_words else 'this'} teach you about your career goals?",
                                "How does that shape what you're looking for in your next role?",
                                "What skills from that experience are you most eager to apply going forward?"
                            ]
                        else:
                            responses = [
                                "Where do you see yourself taking this expertise in the next 2-3 years?",
                                "What aspect of your work energizes you most?",
                                f"How does {key_words[0] if key_words else 'this'} align with your long-term vision?"
                            ]
                        return random.choice(responses)
                
                ai_response = generate_intelligent_fallback(interview_type, answer_characteristics, request.content, conversation_count)
                logger.info(f"Generated intelligent varied fallback response")
        
        # Save AI response
        ai_message = Message(
            id=str(uuid.uuid4()),
            interview_id=interview_id,
            role="assistant",
            content=ai_response,
            is_voice=False
        )
        
        ai_msg_dict = ai_message.model_dump()
        ai_msg_dict['timestamp'] = ai_msg_dict['timestamp'].isoformat()
        await db.messages.insert_one(ai_msg_dict)

        # Optionally kick off async training to keep the local NLP index fresh.
        # Enable by setting environment variable ENABLE_NLP_TRAINING=1
        if os.environ.get("ENABLE_NLP_TRAINING") == "1" and trainer is not None:
            try:
                asyncio.create_task(trainer.train_from_db(db))
            except Exception:
                logger.warning("Failed to schedule background NLP training")

        # Convert to dict and serialize datetime to ISO format for JSON response
        user_response = user_message.model_dump()
        user_response['timestamp'] = user_response['timestamp'].isoformat()
        
        ai_response_dict = ai_message.model_dump()
        ai_response_dict['timestamp'] = ai_response_dict['timestamp'].isoformat()
        
        return {
            "user_message": user_response,
            "ai_message": ai_response_dict
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_router.post("/interviews/{interview_id}/complete")
async def complete_interview(interview_id: str, session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    user = await get_current_user(session_token, authorization)
    
    interview = await db.interview_sessions.find_one(
        {"id": interview_id, "user_id": user.id}
    )
    
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    # Check if already completed - if so, just return success (idempotent operation)
    if interview.get('status') == 'completed' or interview.get('status') == 'evaluated':
        logger.info(f"Interview {interview_id} already completed, returning success")
        return {"message": "Interview already completed", "status": interview.get('status')}
    
    # Mark interview as completed
    await db.interview_sessions.update_one(
        {"id": interview_id},
        {"$set": {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    logger.info(f"Interview {interview_id} marked as completed")
    
    return {"message": "Interview completed successfully", "status": "completed"}

@api_router.post("/interviews/{interview_id}/evaluate")
async def evaluate_interview(interview_id: str, session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    try:
        user = await get_current_user(session_token, authorization)
        
        interview = await db.interview_sessions.find_one(
            {"id": interview_id, "user_id": user.id},
            {"_id": 0}
        )
        
        if not interview:
            raise HTTPException(status_code=404, detail="Interview not found")
        
        # Get all messages
        messages = await db.messages.find(
            {"interview_id": interview_id},
            {"_id": 0}
        ).sort("timestamp", 1).to_list(1000)
        
        if not messages or len(messages) < 2:
            raise HTTPException(status_code=400, detail="Not enough messages to evaluate. Please have at least one exchange.")
        
        # Build conversation for evaluation
        conversation = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in messages
        ])
        
        # Generate evaluation using GPT
        eval_prompt = f"""Analyze this interview conversation and provide a comprehensive evaluation.

Interview Type: {interview['interview_type']}

Conversation:
{conversation}

Provide a detailed evaluation in the following JSON format:
{{
    "overall_score": <float between 0-100>,
    "communication_score": <float between 0-100>,
    "technical_score": <float between 0-100>,
    "problem_solving_score": <float between 0-100>,
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "areas_for_improvement": ["area 1", "area 2", "area 3"],
    "detailed_feedback": "<comprehensive feedback paragraph>"
}}

Note: Scores should be out of 100. Be fair but constructive in your evaluation.
Provide ONLY the JSON response, no additional text."""
        
        # Check if OpenAI client is available
        if not openai_client:
            logger.error("OpenAI client not configured for evaluation")
            raise HTTPException(status_code=500, detail="AI service not configured. Please set OPENAI_API_KEY or EMERGENT_LLM_KEY.")
        
        # Generate evaluation using OpenAI
        logger.info(f"Generating evaluation for interview {interview_id}")
        
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert interview evaluator. Provide detailed, constructive feedback in JSON format."},
                    {"role": "user", "content": eval_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            eval_response = response.choices[0].message.content
            logger.info("Successfully generated evaluation from AI")
            
        except Exception as ai_error:
            logger.error(f"AI evaluation failed: {str(ai_error)}")
            raise HTTPException(status_code=500, detail=f"Failed to generate AI evaluation: {str(ai_error)}")
        
        # Parse evaluation response
        import json
        try:
            eval_data = json.loads(eval_response)
        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to parse AI response as JSON: {eval_response}")
            raise HTTPException(status_code=500, detail="AI returned invalid response format")
        
        # Validate required fields
        required_fields = ['overall_score', 'communication_score', 'technical_score', 
                          'problem_solving_score', 'strengths', 'areas_for_improvement', 'detailed_feedback']
        missing_fields = [field for field in required_fields if field not in eval_data]
        if missing_fields:
            logger.error(f"AI response missing fields: {missing_fields}")
            raise HTTPException(status_code=500, detail=f"AI response missing required fields: {missing_fields}")
        
        evaluation = Evaluation(
            id=str(uuid.uuid4()),
            interview_id=interview_id,
            overall_score=eval_data['overall_score'],
            communication_score=eval_data['communication_score'],
            technical_score=eval_data['technical_score'],
            problem_solving_score=eval_data['problem_solving_score'],
            strengths=eval_data['strengths'],
            areas_for_improvement=eval_data['areas_for_improvement'],
            detailed_feedback=eval_data['detailed_feedback']
        )
        
        eval_dict = evaluation.model_dump()
        eval_dict['created_at'] = eval_dict['created_at'].isoformat()
        await db.evaluations.insert_one(eval_dict)
        
        logger.info(f"Evaluation saved for interview {interview_id}")
        
        # Update interview status
        await db.interview_sessions.update_one(
            {"id": interview_id},
            {"$set": {"status": "evaluated"}}
        )
        
        # Return evaluation with ISO formatted datetime
        eval_response = evaluation.model_dump()
        eval_response['created_at'] = eval_response['created_at'].isoformat()
        
        return eval_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in evaluate_interview: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@api_router.get("/interviews/{interview_id}/evaluation")
async def get_evaluation(interview_id: str, session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    user = await get_current_user(session_token, authorization)
    
    interview = await db.interview_sessions.find_one(
        {"id": interview_id, "user_id": user.id}
    )
    
    if not interview:
        raise HTTPException(status_code=404, detail="Interview not found")
    
    evaluation = await db.evaluations.find_one(
        {"interview_id": interview_id},
        {"_id": 0}
    )
    
    if not evaluation:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    # Ensure datetime fields are in ISO format for JSON serialization
    if isinstance(evaluation.get('created_at'), datetime):
        evaluation['created_at'] = evaluation['created_at'].isoformat()
    
    return evaluation


# --- Optional NLP endpoints -----------------------------------------
@api_router.post("/nlp/train")
async def trigger_nlp_train(session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    """Trigger an on-demand NLP training run (build local embedding index).
    Requires authentication and the `sentence-transformers` + `scikit-learn` packages.
    """
    if trainer is None:
        raise HTTPException(status_code=500, detail="NLP trainer not available: missing dependencies or import error")
    # Require auth
    await get_current_user(session_token, authorization)
    result = await trainer.train_from_db(db)
    return result


@api_router.post("/nlp/predict")
async def nlp_predict(payload: Dict[str, Any], session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
    """Return top historical assistant replies for a given input text.

    Body: { "text": "...", "top_k": 1 }
    """
    if trainer is None:
        raise HTTPException(status_code=500, detail="NLP trainer not available: missing dependencies or import error")
    # Require auth
    await get_current_user(session_token, authorization)
    text = payload.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="Missing text in payload")
    top_k = int(payload.get("top_k", 1))
    return trainer.predict_reply(text, top_k=top_k)

# Configure CORS origins BEFORE adding middleware
cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')
cors_origins = [origin.strip() for origin in cors_origins]  # Remove whitespace

# Add CORS Middleware FIRST (before including router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Add Session Middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get('SESSION_SECRET', 'your-secret-key-change-in-production')
)

# Include router AFTER middleware
app.include_router(api_router)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting AI Interview Assistant Backend")
    logger.info(f"CORS origins: {cors_origins}")
    if openai_client:
        logger.info("✓ OpenAI client configured successfully")
    else:
        logger.warning("✗ OpenAI client not configured - AI features will not work!")
    # Check MongoDB connection if client was created
    if client is not None:
        try:
            # Use the ping command to verify connectivity
            await client.admin.command('ping')
            logger.info("✓ Connected to MongoDB")
        except Exception as e:
            logger.warning(f"✗ Unable to reach MongoDB at {mongo_url}: {e}")
            logger.warning("  Some features requiring DB access will fail until MongoDB is available.")
    else:
        logger.warning("✗ MongoDB client not configured - database features will not work!")
    logger.info("Server is ready to accept requests")

@app.on_event("shutdown")
async def shutdown_db_client():
    logger.info("Shutting down AI Interview Assistant Backend")
    client.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)