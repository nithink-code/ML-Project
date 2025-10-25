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
        "technical": "Hello! I'm your AI interviewer for today's technical interview. I'll be asking you questions about programming, algorithms, data structures, and problem-solving. Let's start with: Can you tell me about your experience with software development?",
        "behavioral": "Hello! I'm your AI interviewer for this behavioral interview. I'll ask questions about your work experience, teamwork, and how you handle various situations. Let's begin: Tell me about a challenging project you worked on and how you approached it.",
        "general": "Hello! Welcome to your interview session. I'll be asking you a variety of questions to understand your background, skills, and experiences. Let's start: Could you give me a brief introduction about yourself and your career goals?"
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
            "technical": "You are an experienced technical interviewer. Ask relevant technical questions about programming, algorithms, data structures, system design, and problem-solving. Provide follow-up questions based on responses. Be professional but encouraging.",
            "behavioral": "You are an experienced HR interviewer conducting a behavioral interview. Ask questions about past experiences, teamwork, conflict resolution, leadership, and how candidates handle challenges. Use the STAR method (Situation, Task, Action, Result) framework.",
            "general": "You are a professional interviewer. Ask a mix of questions about the candidate's background, skills, experience, and career goals. Be conversational and professional."
        }
        
        # Get conversation history
        messages_cursor = db.messages.find({"interview_id": interview_id}).sort("timestamp", 1)
        conversation_history = []
        async for msg in messages_cursor:
            conversation_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Check if OpenAI client is available
        if not openai_client:
            logger.error("OpenAI client not configured")
            # Use fallback mock response for development/testing
            logger.warning("Using mock AI response - configure OPENAI_API_KEY or EMERGENT_LLM_KEY for real AI")
            mock_responses = [
                "That's an interesting point. Could you elaborate on your experience with that?",
                "Great answer! Let me ask you another question: How do you approach problem-solving in challenging situations?",
                "I see. Can you provide a specific example from your past experience?",
                "Thank you for sharing that. What would you say are your key strengths in this area?",
                "Interesting perspective. How do you handle feedback and criticism?",
                "That's good to know. Can you walk me through your thought process when tackling complex problems?"
            ]
            # Simple selection based on message count
            messages_count = len(conversation_history)
            ai_response = mock_responses[messages_count % len(mock_responses)]
            logger.info("Generated mock AI response")
        else:
            # Generate AI response using OpenAI
            messages_for_api = [
                {"role": "system", "content": system_messages.get(interview_type, system_messages["general"])}
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
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    ai_response = response.choices[0].message.content
                    logger.info(f"Successfully generated response using {model}")
                    break  # Success, exit the loop
                except Exception as e:
                    logger.warning(f"Failed with model {model}: {str(e)}")
                    last_error = e
                    continue  # Try next model
            
            if not ai_response:
                logger.error(f"All models failed. Last error: {str(last_error)}")
                # Fallback to mock response instead of failing
                logger.warning("All AI models failed, using mock response")
                mock_responses = [
                    "That's an interesting point. Could you elaborate on your experience with that?",
                    "Great answer! Let me ask you another question: How do you approach problem-solving in challenging situations?",
                    "I see. Can you provide a specific example from your past experience?",
                ]
                messages_count = len(conversation_history)
                ai_response = mock_responses[messages_count % len(mock_responses)]
        
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
    
    await db.interview_sessions.update_one(
        {"id": interview_id},
        {"$set": {
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat()
        }}
    )
    
    return {"message": "Interview completed"}

@api_router.post("/interviews/{interview_id}/evaluate")
async def evaluate_interview(interview_id: str, session_token: Optional[str] = Cookie(None), authorization: Optional[str] = Header(None)):
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
    
    # Build conversation for evaluation
    conversation = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" 
        for msg in messages
    ])
    
    # Generate evaluation using GPT-5
    eval_prompt = f"""Analyze this interview conversation and provide a comprehensive evaluation.

Interview Type: {interview['interview_type']}

Conversation:
{conversation}

Provide a detailed evaluation in the following JSON format:
{{
    "overall_score": <float between 0-10>,
    "communication_score": <float between 0-10>,
    "technical_score": <float between 0-10>,
    "problem_solving_score": <float between 0-10>,
    "strengths": ["strength 1", "strength 2", "strength 3"],
    "areas_for_improvement": ["area 1", "area 2", "area 3"],
    "detailed_feedback": "<comprehensive feedback paragraph>"
}}

Provide ONLY the JSON response, no additional text."""
    
    # Check if OpenAI client is available
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    # Generate evaluation using OpenAI
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
    
    # Parse evaluation response
    import json
    eval_data = json.loads(eval_response)
    
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
    
    # Update interview status
    await db.interview_sessions.update_one(
        {"id": interview_id},
        {"$set": {"status": "evaluated"}}
    )
    
    # Return evaluation with ISO formatted datetime
    eval_response = evaluation.model_dump()
    eval_response['created_at'] = eval_response['created_at'].isoformat()
    
    return eval_response

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
cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
cors_origins = [origin.strip() for origin in cors_origins]  # Remove whitespace

# Add CORS Middleware FIRST (before including router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
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