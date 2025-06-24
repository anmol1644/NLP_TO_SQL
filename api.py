import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
import json
import time
from bson import ObjectId

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
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
        "http://localhost",
        "http://localhost:8080",
        "https://yourproductiondomain.com",  # Add your production domain when ready
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
    expose_headers=["Content-Length", "Content-Range"],
    max_age=3600,  # Cache preflight requests for 1 hour
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
async def create_workspace(workspace: WorkspaceCreate):
    """Create a new workspace"""
    # Default to public user ID
    user_id = ObjectId()
    return await WorkspaceService.create_workspace(workspace, user_id)


@app.get("/workspaces", response_model=List[Workspace])
async def get_user_workspaces():
    """Get all workspaces for the current user"""
    return await WorkspaceService.get_all_workspaces()


@app.get("/workspaces/{workspace_id}", response_model=Workspace)
async def get_workspace(workspace_id: str):
    """Get a workspace by ID"""
    workspace = await WorkspaceService.get_workspace_by_id(workspace_id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    return workspace


@app.put("/workspaces/{workspace_id}", response_model=Workspace)
async def update_workspace(
    workspace_id: str,
    workspace_data: Dict[str, Any] = Body(...)
):
    """Update a workspace"""
    workspace = await WorkspaceService.update_workspace_by_id(
        workspace_id, workspace_data
    )
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    return workspace


@app.delete("/workspaces/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(workspace_id: str):
    """Delete a workspace"""
    deleted = await WorkspaceService.delete_workspace_by_id(workspace_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )


@app.post("/workspaces/{workspace_id}/activate")
async def activate_workspace(workspace_id: str):
    """Activate a workspace by establishing a database connection"""
    try:
        workspace = await WorkspaceService.get_workspace_by_id(workspace_id)
        
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
                "user_id": workspace.user_id,
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workspace activation error: {str(e)}"
        )


@app.post("/workspaces/{workspace_id}/deactivate")
async def deactivate_workspace(workspace_id: str):
    """Deactivate a workspace by closing its database connection"""
    try:
        # Check if workspace exists
        workspace = await WorkspaceService.get_workspace_by_id(workspace_id)
        
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found"
            )
        
        # Check if workspace is active
        if workspace_id not in active_workspaces:
            return {
                "success": True,
                "message": f"Workspace {workspace_id} is already inactive",
                "workspace_id": workspace_id
            }
        
        # Close connection pool
        db_connection_manager.close_workspace_pool(workspace_id)
        
        # Remove from active workspaces
        del active_workspaces[workspace_id]
        
        return {
            "success": True,
            "message": f"Successfully deactivated workspace {workspace_id}",
            "workspace_id": workspace_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workspace deactivation error: {str(e)}"
        )


@app.get("/workspaces/{workspace_id}/status")
async def get_workspace_status(workspace_id: str):
    """Get the connection status of a workspace"""
    try:
        # Check if workspace exists
        workspace = await WorkspaceService.get_workspace_by_id(workspace_id)
        
        if not workspace:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workspace not found"
            )
        
        # Check if workspace is active
        if workspace_id in active_workspaces:
            status_info = active_workspaces[workspace_id]
            return {
                "status": status_info["status"],
                "connected_at": status_info["connected_at"],
                "database_info": status_info["database_info"]
            }
        else:
            return {
                "status": "disconnected",
                "database_info": {
                    "name": workspace.db_connection.db_name,
                    "host": workspace.db_connection.host,
                    "port": workspace.db_connection.port
                }
            }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting workspace status: {str(e)}"
        )


# Session endpoints
@app.post("/workspaces/{workspace_id}/sessions", response_model=Session, status_code=status.HTTP_201_CREATED)
async def create_session(workspace_id: str, session: SessionCreate):
    """Create a new session in a workspace"""
    return await SessionService.create_session(session, workspace_id)


@app.get("/workspaces/{workspace_id}/sessions", response_model=List[Session])
async def get_workspace_sessions(workspace_id: str):
    """Get all sessions in a workspace"""
    return await SessionService.get_workspace_sessions(workspace_id)


@app.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str):
    """Get a session by ID"""
    session = await SessionService.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return session


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(session_id: str):
    """Delete a session"""
    success = await SessionService.delete_session(session_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Clean up the generator for this session
    if session_id in active_generators:
        del active_generators[session_id]
    
    return None


# Message endpoints
@app.get("/sessions/{session_id}/messages", response_model=List[Message])
async def get_session_messages(session_id: str):
    """Get all messages in a session"""
    return await MessageService.get_session_messages(session_id)


@app.get("/messages/{message_id}/query-result")
async def get_message_query_result(message_id: str):
    """Get the query result for a message"""
    message = await MessageService.get_message(message_id)
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    if not message.query_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No query result for this message"
        )
    
    return message.query_result


@app.get("/messages/{message_id}/results")
async def get_message_results(message_id: str, page: int = 1, page_size: int = 10):
    """Get paginated results for a message's query"""
    message = await MessageService.get_message(message_id)
    
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Message not found"
        )
    
    if not message.query_result or not message.query_result.results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results for this message"
        )
    
    # Calculate pagination
    all_results = message.query_result.results
    total_rows = len(all_results)
    total_pages = (total_rows + page_size - 1) // page_size
    
    # Adjust page if out of range
    if page < 1:
        page = 1
    elif page > total_pages and total_pages > 0:
        page = total_pages
    
    # Get slice of results for the requested page
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_rows)
    
    page_results = all_results[start_idx:end_idx]
    
    # Create pagination info
    pagination = {
        "current_page": page,
        "total_pages": total_pages,
        "total_rows": total_rows,
        "page_size": page_size,
        "has_next": page < total_pages,
        "has_prev": page > 1
    }
    
    return {
        "pagination": pagination,
        "data": page_results
    }


@app.post("/sessions/{session_id}/query")
async def query_with_session(
    session_id: str, 
    query_req: QueryRequest
):
    """Execute a query using an existing session"""
    # Get session first
    session = await SessionService.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get workspace info
    workspace_id = session.workspace_id
    workspace = await WorkspaceService.get_workspace_by_id(workspace_id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    # Check if workspace is active and has a connection
    if workspace_id not in active_workspaces:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Workspace {workspace_id} is not active. Please activate it first."
        )
    
    # Get or create SQL generator for this session
    if session_id in active_generators:
        sql_generator = active_generators[session_id]
    else:
        # Create a new SQL generator for this session
        db_analyzer = active_workspaces[workspace_id].get("db_analyzer")
        
        if not db_analyzer:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Database analyzer not found for this workspace"
            )
        
        sql_generator = SmartSQLGenerator(
            db_analyzer=db_analyzer,
            use_memory=True,
            use_cache=True,
            memory_persist_dir=f"./memory_store/{session_id}"
        )
        active_generators[session_id] = sql_generator
    
    # Create user message
    user_message = MessageCreate(
        content=query_req.question,
        role="user",
        session_id=session_id
    )
    user_message = await MessageService.create_message(user_message, session.user_id)
    
    # Execute query
    start_time = time.time()
    auto_fix = query_req.auto_fix if hasattr(query_req, 'auto_fix') else True
    max_attempts = query_req.max_attempts if hasattr(query_req, 'max_attempts') else 2
    
    try:
        query_result = sql_generator.execute_query(
            query_req.question, 
            auto_fix=auto_fix,
            max_attempts=max_attempts
        )
        
        # Update query_result with additional metadata
        execution_time = time.time() - start_time
        query_result["execution_time"] = execution_time
        
        # Create assistant message with query result
        assistant_message = MessageCreate(
            content=query_result.get("text", "Query executed"),
            role="assistant",
            session_id=session_id,
            query_result=query_result
        )
        assistant_message = await MessageService.create_message(
            assistant_message, session.user_id
        )
        
        # Format the response
        response = format_query_result(query_result)
        
        # Add session and message IDs to response
        response["session_id"] = session_id
        response["message_id"] = str(assistant_message.id)
        
        return response
        
    except Exception as e:
        # Handle error and create error message
        error_message = f"Error executing query: {str(e)}"
        error_result = {
            "success": False,
            "error": str(e),
            "text": error_message,
            "execution_time": time.time() - start_time
        }
        
        # Create assistant message with error info
        error_assistant_message = MessageCreate(
            content=error_message,
            role="assistant",
            session_id=session_id,
            query_result=error_result
        )
        await MessageService.create_message(error_assistant_message, session.user_id)
        
        # Return error result
        return error_result


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
    page_size: int = 10
):
    """Get paginated results for a table in a session"""
    # Get session first
    session = await SessionService.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    # Get the SQL generator for this session
    sql_generator = get_sql_generator(session_id)
    
    # Get the results for the specified table and page
    try:
        results = sql_generator.get_paginated_results(table_id, page, page_size)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching paginated results: {str(e)}"
        )


# Workspace connection pool endpoint
@app.get("/admin/connection-pools")
async def get_all_connection_pools():
    """Get information about all database connection pools"""
    return db_connection_manager.get_all_pools_info()


@app.post("/workspaces/{workspace_id}/refresh-schema")
async def refresh_workspace_schema(workspace_id: str):
    """Refresh the database schema for a workspace"""
    # Check if workspace exists
    workspace = await WorkspaceService.get_workspace_by_id(workspace_id)
    
    if not workspace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found"
        )
    
    try:
        # Check if workspace is active
        if workspace_id not in active_workspaces:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Workspace {workspace_id} is not active. Please activate it first."
            )
        
        # Get the database analyzer and refresh schema
        db_analyzer = active_workspaces[workspace_id].get("db_analyzer")
        
        if not db_analyzer:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Database analyzer not found for workspace {workspace_id}"
            )
        
        # Refresh schema
        schema_info = db_analyzer.analyze_schema(force_refresh=True)
        
        # Calculate some summary info
        table_count = len(schema_info.get("tables", {}))
        relationship_count = len(schema_info.get("relationships", []))
        
        return {
            "success": True,
            "message": f"Schema refreshed successfully. Found {table_count} tables and {relationship_count} relationships.",
            "workspace_id": workspace_id,
            "tables_count": table_count,
            "relationships_count": relationship_count,
            "refreshed_at": datetime.utcnow().isoformat()
        }
        
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