import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from smart_sql import SmartSQLGenerator
from db_analyzer import DatabaseAnalyzer


app = FastAPI(
    title="NLP to SQL API",
    description="API for natural language to SQL conversion with multi-user session support",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active user sessions
# Maps session_id -> SmartSQLGenerator instance
active_sessions: Dict[str, Dict[str, Any]] = {}

# Session cleanup interval (in minutes)
SESSION_TIMEOUT = 60  # 1 hour


class QueryRequest(BaseModel):
    question: str
    auto_fix: bool = True
    max_attempts: int = 2


class SessionRequest(BaseModel):
    db_name: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = "localhost"
    port: Optional[str] = "5432"
    use_memory: bool = True
    use_cache: bool = True


def get_sql_generator(session_id: str):
    """Get SQL generator instance for the given session or raise 404 if not found"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired"
        )
    
    # Update last access time
    active_sessions[session_id]["last_access"] = datetime.now()
    
    return active_sessions[session_id]["generator"]


def cleanup_sessions():
    """Remove expired sessions"""
    now = datetime.now()
    expired_sessions = [
        session_id
        for session_id, session_data in active_sessions.items()
        if now - session_data["last_access"] > timedelta(minutes=SESSION_TIMEOUT)
    ]
    
    for session_id in expired_sessions:
        del active_sessions[session_id]
    
    return len(expired_sessions)


@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "NLP to SQL API is running"}


@app.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(session_req: SessionRequest):
    """Create a new session with database connection"""
    # Clean up expired sessions
    cleanup_sessions()
    
    try:
        # Get database connection parameters
        db_name = session_req.db_name or os.getenv("DB_NAME", "postgres")
        username = session_req.username or os.getenv("DB_USERNAME", "postgres")
        password = session_req.password or os.getenv("DB_PASSWORD", "akshwalia")
        host = session_req.host or os.getenv("DB_HOST", "localhost")
        port = session_req.port or os.getenv("DB_PORT", "5432")

        print(db_name, username, password, host, port)
        
        # Initialize database analyzer
        db_analyzer = DatabaseAnalyzer(db_name, username, password, host, port)
        
        # Create a memory directory specific to this session
        session_id = str(uuid.uuid4())
        memory_dir = f"./memory_store/{session_id}"
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize SQL generator
        sql_generator = SmartSQLGenerator(
            db_analyzer,
            use_cache=session_req.use_cache,
            use_memory=session_req.use_memory,
            memory_persist_dir=memory_dir
        )
        
        # Store session
        active_sessions[session_id] = {
            "generator": sql_generator,
            "created_at": datetime.now(),
            "last_access": datetime.now(),
            "db_info": {
                "db_name": db_name,
                "host": host,
                "port": port
            }
        }
        
        return {
            "session_id": session_id,
            "message": "Session created successfully",
            "expires_in_minutes": SESSION_TIMEOUT
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@app.get("/sessions/{session_id}")
def get_session_info(session_id: str):
    """Get information about a session"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired"
        )
    
    session_data = active_sessions[session_id]
    
    # Update last access time
    session_data["last_access"] = datetime.now()
    
    # Calculate remaining time
    elapsed = datetime.now() - session_data["last_access"]
    remaining_minutes = SESSION_TIMEOUT - int(elapsed.total_seconds() / 60)
    
    return {
        "session_id": session_id,
        "created_at": session_data["created_at"].isoformat(),
        "last_access": session_data["last_access"].isoformat(),
        "db_info": session_data["db_info"],
        "expires_in_minutes": max(0, remaining_minutes)
    }


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Delete a session"""
    if session_id not in active_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired"
        )
    
    # Delete the session
    del active_sessions[session_id]
    
    return {"message": f"Session {session_id} deleted successfully"}


@app.post("/query")
def query_without_session(query_req: QueryRequest):
    """Process a query without a session - creates temporary session"""
    try:
        # Create a temporary session
        db_name = os.getenv("DB_NAME", "postgres")
        username = os.getenv("DB_USERNAME", "postgres")
        password = os.getenv("DB_PASSWORD", "postgres")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        
        # Initialize database analyzer
        db_analyzer = DatabaseAnalyzer(db_name, username, password, host, port)
        
        # Initialize SQL generator (without persistent memory)
        sql_generator = SmartSQLGenerator(
            db_analyzer,
            use_cache=True,
            use_memory=False  # No memory for sessionless queries
        )
        
        # Execute the query
        result = sql_generator.execute_query(
            query_req.question,
            auto_fix=query_req.auto_fix,
            max_attempts=query_req.max_attempts
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@app.post("/sessions/{session_id}/query")
def query_with_session(session_id: str, query_req: QueryRequest):
    """Process a query within a session"""
    # Get the SQL generator for this session
    sql_generator = get_sql_generator(session_id)
    
    try:
        # Execute the query
        result = sql_generator.execute_query(
            query_req.question,
            auto_fix=query_req.auto_fix,
            max_attempts=query_req.max_attempts
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@app.get("/sessions")
def list_sessions():
    """List all active sessions"""
    # Clean up expired sessions first
    expired_count = cleanup_sessions()
    
    sessions = []
    for session_id, session_data in active_sessions.items():
        elapsed = datetime.now() - session_data["last_access"]
        remaining_minutes = SESSION_TIMEOUT - int(elapsed.total_seconds() / 60)
        
        sessions.append({
            "session_id": session_id,
            "created_at": session_data["created_at"].isoformat(),
            "last_access": session_data["last_access"].isoformat(),
            "db_info": session_data["db_info"],
            "expires_in_minutes": max(0, remaining_minutes)
        })
    
    return {
        "sessions": sessions,
        "total": len(sessions),
        "expired_removed": expired_count
    }


if __name__ == "__main__":
    # Create memory store directory if it doesn't exist
    os.makedirs("./memory_store", exist_ok=True)
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000) 