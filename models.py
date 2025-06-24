import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from decimal import Decimal
from bson import ObjectId
from pydantic import BaseModel, Field, EmailStr, validator, field_validator
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB connection
try:
    MONGODB_URL = os.getenv("MONGODB_URL")
    if not MONGODB_URL:
        logger.warning("MONGODB_URL not found in .env file, using default local connection")
        MONGODB_URL = "mongodb://localhost:27017"
    
    logger.info(f"Connecting to MongoDB using URL: {MONGODB_URL.split('@')[0]}@****")
    client = MongoClient(MONGODB_URL)
    
    # Test connection
    client.admin.command('ping')
    logger.info("Successfully connected to MongoDB")
    
    db = client.get_database("nlp_sql")
    
    # Collections
    users_collection = db.users
    workspaces_collection = db.workspaces
    sessions_collection = db.sessions
    messages_collection = db.messages
    
except Exception as e:
    logger.error(f"Failed to connect to MongoDB: {str(e)}")
    # Create fallback in-memory collections for development
    logger.warning("Using in-memory collections as fallback")
    
    class MockCollection:
        def __init__(self):
            self.data = []
            self._id_counter = 1
        
        def insert_one(self, document):
            document["_id"] = str(ObjectId())
            self.data.append(document)
            return type("InsertOneResult", (), {"inserted_id": document["_id"]})
        
        def find_one(self, query):
            for item in self.data:
                if "_id" in query and item.get("_id") == query["_id"]:
                    return item
                if "email" in query and item.get("email") == query["email"]:
                    return item
            return None
        
        def find(self, query=None):
            if query is None:
                query = {}
            
            results = []
            for item in self.data:
                match = True
                for k, v in query.items():
                    if k not in item or item[k] != v:
                        match = False
                        break
                if match:
                    results.append(item)
            
            return type("Cursor", (), {
                "__iter__": lambda self: iter(results),
                "limit": lambda n: results[:n],
                "count": lambda: len(results)
            })
        
        def update_one(self, query, update):
            for i, item in enumerate(self.data):
                if "_id" in query and item.get("_id") == query["_id"]:
                    if "$set" in update:
                        for k, v in update["$set"].items():
                            self.data[i][k] = v
                    return type("UpdateResult", (), {"modified_count": 1})
            return type("UpdateResult", (), {"modified_count": 0})
        
        def delete_one(self, query):
            for i, item in enumerate(self.data):
                if "_id" in query and item.get("_id") == query["_id"]:
                    del self.data[i]
                    return type("DeleteResult", (), {"deleted_count": 1})
            return type("DeleteResult", (), {"deleted_count": 0})
    
    # Create mock collections
    users_collection = MockCollection()
    workspaces_collection = MockCollection()
    sessions_collection = MockCollection()
    messages_collection = MockCollection()
    
    # Add a default admin user
    default_password = "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW"  # "password"
    users_collection.insert_one({
        "email": "admin@example.com",
        "hashed_password": default_password,
        "first_name": "Admin",
        "is_active": True,
        "is_admin": True,
        "created_at": datetime.utcnow()
    })
    logger.info("Created default user: admin@example.com with password: password")

# Custom ObjectId field for Pydantic models
class PyObjectId(str):
    @classmethod
    def __get_pydantic_core_schema__(cls, _source_type, _handler):
        from pydantic_core import core_schema
        
        def validate_objectid(value):
            if isinstance(value, ObjectId):
                return str(value)
            if not ObjectId.is_valid(value):
                raise ValueError("Invalid ObjectId")
            return str(value)
        
        return core_schema.no_info_plain_validator_function(
            function=validate_objectid,
            serialization=core_schema.to_string_ser_schema(),
        )


# User models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None


class User(BaseModel):
    id: PyObjectId = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    email: EmailStr
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool = True
    is_admin: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "_id": "123456789012345678901234",
                "email": "user@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "is_active": True
            }
        }


class UserInDB(User):
    hashed_password: str


# Token models
class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: datetime


class TokenData(BaseModel):
    user_id: Optional[str] = None
    exp: Optional[datetime] = None


# Database connection model
class DatabaseConnection(BaseModel):
    db_name: str
    username: str
    password: str
    host: str = "localhost"
    port: int = 5432


# Workspace models
class WorkspaceCreate(BaseModel):
    name: str
    description: Optional[str] = None
    db_connection: DatabaseConnection


class Workspace(BaseModel):
    id: PyObjectId = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    name: str
    description: Optional[str] = None
    db_connection: DatabaseConnection
    user_id: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class WorkspaceInDB(Workspace):
    pass


# Session models
class SessionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    workspace_id: PyObjectId


class Session(BaseModel):
    id: PyObjectId = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    name: str
    description: Optional[str] = None
    workspace_id: PyObjectId
    user_id: PyObjectId
    vector_store_id: str = Field(default_factory=lambda: str(ObjectId()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class SessionInDB(Session):
    pass


# Query execution result models
class QueryResult(BaseModel):
    success: bool
    sql: Optional[str] = None
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    is_conversational: Optional[bool] = None
    is_multi_query: Optional[bool] = None
    is_why_analysis: Optional[bool] = None
    query_type: Optional[str] = None
    analysis_type: Optional[str] = None
    source: Optional[str] = None
    confidence: Optional[int] = None
    auto_fixed: Optional[bool] = None
    fix_attempts: Optional[int] = None
    pagination: Optional[Dict[str, Any]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    
    @field_validator('results', 'pagination', 'tables', mode='before')
    @classmethod
    def convert_decimals(cls, v):
        """Convert Decimal objects to float for JSON serialization"""
        if v is None:
            return v
        return cls._convert_decimals_recursive(v)
    
    @staticmethod
    def _convert_decimals_recursive(obj):
        """Recursively convert Decimal objects to float"""
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: QueryResult._convert_decimals_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [QueryResult._convert_decimals_recursive(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(QueryResult._convert_decimals_recursive(item) for item in obj)
        else:
            return obj


# Message models
class MessageCreate(BaseModel):
    content: str
    role: str  # "user" or "assistant"
    session_id: PyObjectId
    query_result: Optional[QueryResult] = None


class Message(BaseModel):
    id: PyObjectId = Field(default_factory=lambda: str(ObjectId()), alias="_id")
    content: str
    role: str  # "user" or "assistant"
    session_id: PyObjectId
    user_id: PyObjectId
    query_result: Optional[QueryResult] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


class MessageInDB(Message):
    pass