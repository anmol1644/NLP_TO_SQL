import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import json

import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status, Header, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

from smart_sql import SmartSQLGenerator
from db_analyzer import DatabaseAnalyzer
import psycopg2
from psycopg2 import OperationalError
from models import (
    UserCreate, User, Token,
    WorkspaceCreate, Workspace, DatabaseConnection,
    SessionCreate, Session,
    MessageCreate, Message, QueryResult
)
from auth import (
    authenticate_user, create_access_token,
    get_current_active_user, get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from db_service import (
    UserService, WorkspaceService,
    SessionService, MessageService
)
from vector_store import vector_store_manager
from db_connection_manager import db_connection_manager, cleanup_db_connections


app = FastAPI(
    title="NLP to SQL API",
    description="API for natural language to SQL conversion with multi-user support",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active SQL generators for each session
# Maps session_id -> SmartSQLGenerator instance
active_generators: Dict[str, SmartSQLGenerator] = {}

# Active workspace connections
# Maps workspace_id -> connection status and info
active_workspaces: Dict[str, Dict[str, Any]] = {}

# Session cleanup interval (in minutes)
SESSION_TIMEOUT = 60  # 1 hour


def convert_decimals_to_float(obj):
    """Recursively convert Decimal objects to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_decimals_to_float(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_decimals_to_float(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_decimals_to_float(item) for item in obj)
    else:
        return obj


class TableInfo(BaseModel):
    """Information about a table returned in multi-query analysis"""
    name: str = Field(description="The name/title of the table")
    description: str = Field(description="Description of what this table contains")
    sql: str = Field(description="The SQL query used to generate this table")
    results: List[Dict[str, Any]] = Field(description="The query results for this table")


class QueryRequest(BaseModel):
    question: str
    auto_fix: bool = True
    max_attempts: int = 2


class SessionRequest(BaseModel):
    name: str
    description: Optional[str] = None
    workspace_id: str


def get_sql_generator(session_id: str):
    """Get SQL generator instance for the given session or raise 404 if not found"""
    if session_id not in active_generators:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired"
        )
    
    return active_generators[session_id]


def cleanup_inactive_generators():
    """Clean up inactive SQL generators"""
    # This would be better with a background task
    # For now, just a placeholder
    pass


# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "NLP to SQL API is running"}


# Authentication endpoints
@app.post("/auth/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register_user(user: UserCreate):
    """Register a new user"""
    try:
        return await UserService.create_user(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get an access token"""
    user = authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Update last login time
    await UserService.update_last_login(str(user.id))
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.id)},
        expires_delta=access_token_expires
    )
    
    # Calculate expiration time for client
    expires_at = datetime.utcnow() + access_token_expires
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": expires_at
    }


@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user


# Workspace endpoints
@app.post("/workspaces", response_model=Workspace, status_code=status.HTTP_201_CREATED)
async def create_workspace(
    workspace: WorkspaceCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new workspace"""
    return await WorkspaceService.create_workspace(workspace, current_user.id)


@app.get("/workspaces", response_model=List[Workspace])
async def get_user_workspaces(current_user: User = Depends(get_current_active_user)):
    """Get all workspaces for the current user"""
    return await WorkspaceService.get_user_workspaces(current_user.id)


@app.get("/workspaces/{workspace_id}", response_model=Workspace)
async def get_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a workspace by ID"""
    workspace = await WorkspaceService.get_workspace(workspace_id, current_user.id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    return workspace


@app.put("/workspaces/{workspace_id}", response_model=Workspace)
async def update_workspace(
    workspace_id: str,
    workspace_data: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user)
):
    """Update a workspace"""
    workspace = await WorkspaceService.update_workspace(
        workspace_id, current_user.id, workspace_data
    )
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    return workspace


@app.delete("/workspaces/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a workspace"""
    success = await WorkspaceService.delete_workspace(workspace_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Close connection pool and remove from active workspaces
    db_connection_manager.close_workspace_pool(workspace_id)
    if workspace_id in active_workspaces:
        del active_workspaces[workspace_id]
    
    return None


@app.post("/workspaces/{workspace_id}/activate")
async def activate_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Activate a workspace by connecting to its database"""
    # Get the workspace
    workspace = await WorkspaceService.get_workspace(workspace_id, current_user.id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    db_conn = workspace.db_connection
    
    try:
        # Create database configuration for connection manager
        db_config = {
            'host': db_conn.host,
            'port': db_conn.port,
            'db_name': db_conn.db_name,
            'username': db_conn.username,
            'password': db_conn.password
        }
        
        # Create connection pool for this workspace
        pool_created = db_connection_manager.create_workspace_pool(workspace_id, db_config)
        
        if not pool_created:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create database connection pool"
            )
        
        # Get the database analyzer from connection manager (already created with schema analyzed)
        db_analyzer = db_connection_manager.get_database_analyzer(workspace_id)
        
        # Get table count from the analyzed schema
        table_count = 0
        if db_analyzer and db_analyzer.schema_info and "tables" in db_analyzer.schema_info:
            table_count = len(db_analyzer.schema_info["tables"])
        
        # Store the active workspace connection info
        active_workspaces[workspace_id] = {
            "status": "connected",
            "connected_at": datetime.utcnow().isoformat(),
            "user_id": current_user.id,
            "database_info": {
                "name": db_conn.db_name,
                "host": db_conn.host,
                "port": db_conn.port,
                "table_count": table_count
            },
            "db_analyzer": db_analyzer
        }
        
        return {
            "success": True,
            "message": f"Successfully connected to database '{db_conn.db_name}'",
            "workspace_id": workspace_id,
            "database_info": {
                "name": db_conn.db_name,
                "host": db_conn.host,
                "port": db_conn.port,
                "table_count": table_count,
                "status": "connected"
            }
        }
        
    except OperationalError as e:
        error_msg = str(e).strip()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to connect to database: {error_msg}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database connection error: {str(e)}"
        )


@app.post("/workspaces/{workspace_id}/deactivate")
async def deactivate_workspace(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Deactivate a workspace by disconnecting from its database"""
    # Check if workspace exists and belongs to user
    workspace = await WorkspaceService.get_workspace(workspace_id, current_user.id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Close the connection pool
    pool_closed = db_connection_manager.close_workspace_pool(workspace_id)
    
    # Remove from active workspaces
    if workspace_id in active_workspaces:
        del active_workspaces[workspace_id]
    
    return {
        "success": True,
        "message": "Workspace deactivated successfully",
        "workspace_id": workspace_id,
        "status": "disconnected",
        "pool_closed": pool_closed
    }


@app.get("/workspaces/{workspace_id}/status")
async def get_workspace_status(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the connection status of a workspace"""
    # Check if workspace exists and belongs to user
    workspace = await WorkspaceService.get_workspace(workspace_id, current_user.id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Check connection manager status first
    pool_status = db_connection_manager.get_workspace_status(workspace_id)
    
    if pool_status:
        return pool_status
    else:
        return {
            "workspace_id": workspace_id,
            "status": "disconnected",
            "database_info": {
                "name": workspace.db_connection.db_name,
                "host": workspace.db_connection.host,
                "port": workspace.db_connection.port,
                "status": "disconnected"
            }
        }


# Session endpoints
@app.post("/workspaces/{workspace_id}/sessions", response_model=Session, status_code=status.HTTP_201_CREATED)
async def create_session(
    workspace_id: str,
    session: SessionCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new session in a workspace"""
    # Make sure the session has the correct workspace_id
    session_data = session.dict()
    session_data["workspace_id"] = workspace_id
    session = SessionCreate(**session_data)
    
    try:
        return await SessionService.create_session(session, current_user.id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@app.get("/workspaces/{workspace_id}/sessions", response_model=List[Session])
async def get_workspace_sessions(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get all sessions for a workspace"""
    return await SessionService.get_workspace_sessions(workspace_id, current_user.id)


@app.get("/sessions/{session_id}", response_model=Session)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a session by ID"""
    session = await SessionService.get_session(session_id, current_user.id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return session


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a session"""
    success = await SessionService.delete_session(session_id, current_user.id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Clean up any active generator
    if session_id in active_generators:
        del active_generators[session_id]
    
    return None


# Message endpoints
@app.get("/sessions/{session_id}/messages", response_model=List[Message])
async def get_session_messages(
    session_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get all messages for a session including complete query results"""
    return await MessageService.get_session_messages(session_id, current_user.id)


@app.get("/messages/{message_id}/query-result")
async def get_message_query_result(
    message_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the complete query result for a specific message"""
    # Get the message
    message = await MessageService.get_message(message_id, current_user.id)
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    if not message.query_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No query result available for this message"
        )
    
    return message.query_result


@app.get("/messages/{message_id}/results")
async def get_message_results(
    message_id: str,
    page: int = 1,
    page_size: int = 10,
    current_user: User = Depends(get_current_active_user)
):
    """Get paginated results for a message's query result"""
    # Get the message
    message = await MessageService.get_message(message_id, current_user.id)
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    if not message.query_result or not message.query_result.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No query results available for this message"
        )
    
    # Paginate the results
    results = message.query_result.results
    total_rows = len(results)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Validate page number
    if page < 1:
        page = 1
    elif page > total_pages:
        page = total_pages
    
    # Calculate start and end indices
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    # Get the requested page of results
    page_results = results[start_idx:end_idx]
    
    return {
        "success": True,
        "message_id": message_id,
        "sql": message.query_result.sql,
        "results": page_results,
        "pagination": {
            "total_rows": total_rows,
            "page_size": page_size,
            "current_page": page,
            "total_pages": total_pages
        },
        "query_metadata": {
            "execution_time": message.query_result.execution_time,
            "query_type": message.query_result.query_type,
            "is_multi_query": message.query_result.is_multi_query,
            "auto_fixed": message.query_result.auto_fixed
        }
    }


@app.post("/sessions/{session_id}/query")
async def query_with_session(
    session_id: str, 
    query_req: QueryRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Process a query within a session"""
    # Get the session
    session = await SessionService.get_session(session_id, current_user.id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get the workspace
    workspace = await WorkspaceService.get_workspace(session.workspace_id, current_user.id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Initialize or get the SQL generator for this session
    if session_id not in active_generators:
        # Try to get database analyzer from connection manager first
        db_analyzer = db_connection_manager.get_database_analyzer(session.workspace_id)
        
        # Ensure schema is analyzed if we have a database analyzer
        if db_analyzer:
            db_connection_manager.ensure_schema_analyzed(session.workspace_id)
        
        if not db_analyzer:
            # Check if workspace has a connection pool but no analyzer
            pool_status = db_connection_manager.get_workspace_status(session.workspace_id)
            
            if pool_status:
                # This shouldn't happen, but handle it gracefully
                db_analyzer = db_connection_manager.get_database_analyzer(session.workspace_id)
            elif session.workspace_id in active_workspaces:
                # Use the existing database analyzer from active workspace (fallback)
                db_analyzer = active_workspaces[session.workspace_id]["db_analyzer"]
            else:
                # Create connection pool if it doesn't exist
                db_conn = workspace.db_connection
                db_config = {
                    'host': db_conn.host,
                    'port': db_conn.port,
                    'db_name': db_conn.db_name,
                    'username': db_conn.username,
                    'password': db_conn.password
                }
                
                pool_created = db_connection_manager.create_workspace_pool(session.workspace_id, db_config, analyze_schema=True)
                
                if pool_created:
                    # Get the analyzer from the connection manager
                    db_analyzer = db_connection_manager.get_database_analyzer(session.workspace_id)
                else:
                    # Fallback to direct connection
                    db_analyzer = DatabaseAnalyzer(
                        db_conn.db_name,
                        db_conn.username,
                        db_conn.password,
                        db_conn.host,
                        db_conn.port
                    )
                    # Analyze schema for fallback case
                    db_analyzer.analyze_schema()
        
        # Create a memory directory specific to this session
        memory_dir = f"./vector_stores/{session.vector_store_id}"
        os.makedirs(memory_dir, exist_ok=True)
        
        # Initialize SQL generator
        sql_generator = SmartSQLGenerator(
            db_analyzer,
            use_cache=True,
            use_memory=True,
            memory_persist_dir=memory_dir
        )
        
        active_generators[session_id] = sql_generator
    else:
        sql_generator = active_generators[session_id]
    
<<<<<<< HEAD
=======
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Session creation error: {str(e)}\n{error_trace}")
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
>>>>>>> 43ccfc94c426ddc1f708e39297b6f8184ed2ac5a
    try:
        # Create the user message
        user_message = await MessageService.create_message(
            MessageCreate(
                content=query_req.question,
                role="user",
                session_id=session_id
            ),
            current_user.id
        )
        
        # Execute the query
        result = sql_generator.execute_query(
            query_req.question,
            auto_fix=query_req.auto_fix,
            max_attempts=query_req.max_attempts
        )
        
        # Convert any Decimal objects to float for MongoDB compatibility
        converted_result = convert_decimals_to_float(result)
        
        # Create QueryResult object from the execution result
        query_result = QueryResult(
            success=converted_result.get("success", False),
            sql=converted_result.get("sql"),
            results=converted_result.get("results"),
            error=converted_result.get("error"),
            execution_time=converted_result.get("execution_time"),
            is_conversational=converted_result.get("is_conversational"),
            is_multi_query=converted_result.get("is_multi_query"),
            is_why_analysis=converted_result.get("is_why_analysis"),
            query_type=converted_result.get("query_type"),
            analysis_type=converted_result.get("analysis_type"),
            source=converted_result.get("source"),
            confidence=converted_result.get("confidence"),
            auto_fixed=converted_result.get("auto_fixed"),
            fix_attempts=converted_result.get("fix_attempts"),
            pagination=converted_result.get("pagination"),
            tables=converted_result.get("tables")
        )
        
        # Create the assistant message with complete query result
        assistant_message = await MessageService.create_message(
            MessageCreate(
                content=converted_result.get("text", "I couldn't generate a response."),
                role="assistant",
                session_id=session_id,
                query_result=query_result
            ),
            current_user.id
        )
        
        # Format the response
        response = format_query_result(converted_result)
        response["user_message"] = user_message
        response["assistant_message"] = assistant_message
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


def format_query_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Format and standardize query result for API response"""
    response = {
        "success": result.get("success", False),
        "question": result.get("question", ""),
        "text": result.get("text", ""),
        "execution_time": result.get("execution_time", 0),
    }
    
    # Include pagination metadata if available
    if "pagination" in result:
        response["pagination"] = result["pagination"]
    
    # For conversational queries, just return the text
    if result.get("is_conversational", False):
        response["query_type"] = "conversational"
        return response
    
    # For multi-query analysis, format tables with headers
    if result.get("is_multi_query", False) or result.get("is_why_analysis", False):
        response["query_type"] = "analysis"
        response["is_multi_query"] = True  # Ensure flag is set
        
        # Add special flag for "why" questions
        if result.get("is_why_analysis", False):
            response["analysis_type"] = "causal"
        else:
            response["analysis_type"] = "comparative"
        
        # Format tables with proper metadata
        tables = []
        for table_data in result.get("tables", []):
            table = {
                "name": table_data.get("query_name", "Unnamed Table"),
                "description": table_data.get("description", ""),
                "sql": table_data.get("sql", ""),
                "results": table_data.get("results", []),
                "row_count": table_data.get("row_count", len(table_data.get("results", [])))
            }
            tables.append(table)
        
        response["tables"] = tables
        return response
    
    # For standard SQL queries, include the SQL and results
    response["query_type"] = "sql"
    response["sql"] = result.get("sql", "")
    response["results"] = result.get("results", [])
    
    # Include error message if query failed
    if not response["success"]:
        response["error"] = result.get("error", "Unknown error")
    
    return response


@app.get("/sessions/{session_id}/results/{table_id}")
async def get_paginated_results(
    session_id: str,
    table_id: str,
    page: int = 1,
    page_size: int = 10,
    current_user: User = Depends(get_current_active_user)
):
    """Get paginated results for a query"""
    # Check if the session exists and belongs to the user
    session = await SessionService.get_session(session_id, current_user.id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Check if the SQL generator exists for this session
    if session_id not in active_generators:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No active SQL generator for this session"
        )
    
    sql_generator = active_generators[session_id]
    
    try:
        # Get the paginated results
        result = sql_generator.get_paginated_results(table_id, page, page_size)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Table not found")
            )
        
        # Format the response
        return format_query_result(result)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve paginated results: {str(e)}"
        )


# Add connection pool monitoring endpoint
@app.get("/admin/connection-pools")
async def get_all_connection_pools(current_user: User = Depends(get_current_active_user)):
    """Get status of all active connection pools (admin endpoint)"""
    return db_connection_manager.get_all_workspace_status()


@app.post("/workspaces/{workspace_id}/refresh-schema")
async def refresh_workspace_schema(
    workspace_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Refresh the database schema analysis for a workspace"""
    # Check if workspace exists and belongs to user
    workspace = await WorkspaceService.get_workspace(workspace_id, current_user.id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Check if workspace has an active connection pool
    db_analyzer = db_connection_manager.get_database_analyzer(workspace_id)
    
    if not db_analyzer:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workspace is not currently active. Please activate it first."
        )
    
    try:
        # Force re-analysis by clearing the analyzed flag and re-analyzing
        with db_connection_manager._lock:
            if workspace_id in db_connection_manager.workspace_pools:
                db_connection_manager.workspace_pools[workspace_id]['schema_analyzed'] = False
        
        # Trigger schema analysis
        schema_analyzed = db_connection_manager.ensure_schema_analyzed(workspace_id)
        
        if schema_analyzed:
            # Get updated status
            status_info = db_connection_manager.get_workspace_status(workspace_id)
            return {
                "success": True,
                "message": "Schema analysis refreshed successfully",
                "workspace_id": workspace_id,
                "status": status_info
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to refresh schema analysis"
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error refreshing schema: {str(e)}"
        )


# Cleanup function for graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on application shutdown"""
    cleanup_db_connections()


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("./vector_stores", exist_ok=True)
    
    # Run the API server
    uvicorn.run(app, host="0.0.0.0", port=8000) 