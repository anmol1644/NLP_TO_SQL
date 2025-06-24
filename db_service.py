from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId

from models import (
    UserCreate, UserInDB, User,
    WorkspaceCreate, WorkspaceInDB, Workspace,
    SessionCreate, SessionInDB, Session,
    MessageCreate, MessageInDB, Message,
    users_collection, workspaces_collection, sessions_collection, messages_collection
)
from auth import get_password_hash
from vector_store import vector_store_manager


class UserService:
    """Service for user management"""
    
    @staticmethod
    async def create_user(user: UserCreate) -> User:
        """Create a new user"""
        # Check if user with this email already exists
        if users_collection.find_one({"email": user.email}):
            raise ValueError(f"User with email {user.email} already exists")
        
        # Create a new user
        user_in_db = UserInDB(
            **user.model_dump(exclude={"password"}),
            hashed_password=get_password_hash(user.password)
        )
        
        # Insert into database
        result = users_collection.insert_one(user_in_db.model_dump(by_alias=True))
        
        # Get the created user
        created_user = users_collection.find_one({"_id": result.inserted_id})
        
        # Convert ObjectId to string
        created_user["_id"] = str(created_user["_id"])
        
        return User(**created_user)
    
    @staticmethod
    async def get_user(user_id: str) -> Optional[User]:
        """Get a user by ID"""
        user = users_collection.find_one({"_id": user_id})
        
        if not user:
            return None
        
        # Convert ObjectId to string
        user["_id"] = str(user["_id"])
        
        return User(**user)
    
    @staticmethod
    async def update_last_login(user_id: str) -> None:
        """Update a user's last login time"""
        users_collection.update_one(
            {"_id": user_id},
            {"$set": {"last_login": datetime.utcnow()}}
        )


class WorkspaceService:
    """Service for workspace management"""
    
    @staticmethod
    async def create_workspace(workspace: WorkspaceCreate, user_id: str) -> Workspace:
        """Create a new workspace"""
        # Create a new workspace
        workspace_in_db = WorkspaceInDB(
            **workspace.model_dump(),
            user_id=user_id
        )
        
        # Insert into database
        result = workspaces_collection.insert_one(workspace_in_db.model_dump(by_alias=True))
        
        # Get the created workspace
        created_workspace = workspaces_collection.find_one({"_id": result.inserted_id})
        
        # Convert ObjectIds to strings
        created_workspace["_id"] = str(created_workspace["_id"])
        if isinstance(created_workspace["user_id"], ObjectId):
            created_workspace["user_id"] = str(created_workspace["user_id"])
        
        return Workspace(**created_workspace)
    
    @staticmethod
    async def get_workspace(workspace_id: str, user_id: str) -> Optional[Workspace]:
        """Get a workspace by ID"""
        workspace = workspaces_collection.find_one({
            "_id": workspace_id,
            "user_id": user_id
        })
        
        if not workspace:
            return None
        
        # Convert ObjectIds to strings
        workspace["_id"] = str(workspace["_id"])
        if isinstance(workspace["user_id"], ObjectId):
            workspace["user_id"] = str(workspace["user_id"])
        
        return Workspace(**workspace)
    
    @staticmethod
    async def get_user_workspaces(user_id: str) -> List[Workspace]:
        """Get all workspaces for a user"""
        workspaces = list(workspaces_collection.find({"user_id": user_id}))
        
        # Convert ObjectIds to strings
        for workspace in workspaces:
            workspace["_id"] = str(workspace["_id"])
            if isinstance(workspace["user_id"], ObjectId):
                workspace["user_id"] = str(workspace["user_id"])
        
        return [Workspace(**workspace) for workspace in workspaces]
    
    @staticmethod
    async def update_workspace(
        workspace_id: str, 
        user_id: str, 
        workspace_data: Dict[str, Any]
    ) -> Optional[Workspace]:
        """Update a workspace"""
        # Update the workspace
        result = workspaces_collection.update_one(
            {"_id": workspace_id, "user_id": user_id},
            {"$set": {**workspace_data, "updated_at": datetime.utcnow()}}
        )
        
        if result.matched_count == 0:
            return None
        
        # Get the updated workspace
        updated_workspace = workspaces_collection.find_one({"_id": workspace_id})
        
        # Convert ObjectIds to strings
        updated_workspace["_id"] = str(updated_workspace["_id"])
        if isinstance(updated_workspace["user_id"], ObjectId):
            updated_workspace["user_id"] = str(updated_workspace["user_id"])
        
        return Workspace(**updated_workspace)
    
    @staticmethod
    async def delete_workspace(workspace_id: str, user_id: str) -> bool:
        """Delete a workspace"""
        # Delete all sessions in this workspace first
        sessions = list(sessions_collection.find({"workspace_id": workspace_id}))
        
        for session in sessions:
            # Delete the vector store if it exists
            if "vector_store_id" in session and session["vector_store_id"]:
                vector_store_manager.delete_store(session["vector_store_id"])
            
            # Delete the session
            sessions_collection.delete_one({"_id": session["_id"]})
        
        # Delete the workspace
        result = workspaces_collection.delete_one({
            "_id": workspace_id,
            "user_id": user_id
        })
        
        return result.deleted_count > 0


class SessionService:
    """Service for session management"""
    
    @staticmethod
    async def create_session(session: SessionCreate, user_id: str) -> Session:
        """Create a new session"""
        # Check if the workspace exists and belongs to the user
        workspace = workspaces_collection.find_one({
            "_id": session.workspace_id,
            "user_id": user_id
        })
        
        if not workspace:
            raise ValueError("Workspace not found or does not belong to the user")
        
        # Create a vector store for this session
        vector_store_id = vector_store_manager.create_store(str(session.workspace_id))
        
        # Create a new session
        session_in_db = SessionInDB(
            **session.model_dump(),
            user_id=user_id,
            vector_store_id=vector_store_id
        )
        
        # Insert into database
        result = sessions_collection.insert_one(session_in_db.model_dump(by_alias=True))
        
        # Get the created session
        created_session = sessions_collection.find_one({"_id": result.inserted_id})
        
        # Convert ObjectIds to strings
        created_session["_id"] = str(created_session["_id"])
        if isinstance(created_session["workspace_id"], ObjectId):
            created_session["workspace_id"] = str(created_session["workspace_id"])
        if isinstance(created_session["user_id"], ObjectId):
            created_session["user_id"] = str(created_session["user_id"])
        
        return Session(**created_session)
    
    @staticmethod
    async def get_session(session_id: str, user_id: str) -> Optional[Session]:
        """Get a session by ID"""
        session = sessions_collection.find_one({
            "_id": session_id,
            "user_id": user_id
        })
        
        if not session:
            return None
        
        # Convert ObjectIds to strings
        session["_id"] = str(session["_id"])
        if isinstance(session["workspace_id"], ObjectId):
            session["workspace_id"] = str(session["workspace_id"])
        if isinstance(session["user_id"], ObjectId):
            session["user_id"] = str(session["user_id"])
        
        return Session(**session)
    
    @staticmethod
    async def get_workspace_sessions(workspace_id: str, user_id: str) -> List[Session]:
        """Get all sessions for a workspace"""
        sessions = list(sessions_collection.find({
            "workspace_id": workspace_id,
            "user_id": user_id
        }))
        
        # Convert ObjectIds to strings
        for session in sessions:
            session["_id"] = str(session["_id"])
            if isinstance(session["workspace_id"], ObjectId):
                session["workspace_id"] = str(session["workspace_id"])
            if isinstance(session["user_id"], ObjectId):
                session["user_id"] = str(session["user_id"])
        
        return [Session(**session) for session in sessions]
    
    @staticmethod
    async def update_session_activity(session_id: str, user_id: str) -> None:
        """Update a session's last active time"""
        sessions_collection.update_one(
            {"_id": session_id, "user_id": user_id},
            {"$set": {"last_active": datetime.utcnow(), "updated_at": datetime.utcnow()}}
        )
    
    @staticmethod
    async def delete_session(session_id: str, user_id: str) -> bool:
        """Delete a session"""
        # Get the session first to get the vector store ID
        session = sessions_collection.find_one({
            "_id": session_id,
            "user_id": user_id
        })
        
        if not session:
            return False
        
        # Delete the vector store if it exists
        if "vector_store_id" in session and session["vector_store_id"]:
            vector_store_manager.delete_store(session["vector_store_id"])
        
        # Delete the session
        result = sessions_collection.delete_one({
            "_id": session_id,
            "user_id": user_id
        })
        
        return result.deleted_count > 0


class MessageService:
    """Service for message management"""
    
    @staticmethod
    async def create_message(message: MessageCreate, user_id: str) -> Message:
        """Create a new message"""
        # Check if the session exists and belongs to the user
        session = sessions_collection.find_one({
            "_id": message.session_id,
            "user_id": user_id
        })
        
        if not session:
            raise ValueError("Session not found or does not belong to the user")
        
        # Create a new message
        message_in_db = MessageInDB(
            **message.model_dump(),
            user_id=user_id
        )
        
        # Insert into database
        result = messages_collection.insert_one(message_in_db.model_dump(by_alias=True))
        
        # Get the created message
        created_message = messages_collection.find_one({"_id": result.inserted_id})
        
        # Convert ObjectIds to strings
        created_message["_id"] = str(created_message["_id"])
        if isinstance(created_message["session_id"], ObjectId):
            created_message["session_id"] = str(created_message["session_id"])
        if isinstance(created_message["user_id"], ObjectId):
            created_message["user_id"] = str(created_message["user_id"])
        
        # Add to vector store if it exists
        if "vector_store_id" in session and session["vector_store_id"]:
            # Prepare metadata including query result if available
            metadata = {}
            if hasattr(message, 'query_result') and message.query_result:
                metadata.update({
                    "has_query_result": True,
                    "query_success": message.query_result.success,
                    "query_type": message.query_result.query_type,
                    "is_conversational": message.query_result.is_conversational,
                    "is_multi_query": message.query_result.is_multi_query,
                    "sql": message.query_result.sql[:200] if message.query_result.sql else None,  # Truncate SQL for metadata
                })
            
            vector_store_manager.add_message_to_store(
                session["vector_store_id"],
                str(session["_id"]),
                message.content,
                message.role,
                metadata
            )
        
        # Update session activity
        await SessionService.update_session_activity(str(session["_id"]), user_id)
        
        return Message(**created_message)
    
    @staticmethod
    async def get_session_messages(session_id: str, user_id: str) -> List[Message]:
        """Get all messages for a session"""
        # Check if the session exists and belongs to the user
        session = sessions_collection.find_one({
            "_id": session_id,
            "user_id": user_id
        })
        
        if not session:
            return []
        
        # Get messages
        messages = list(messages_collection.find(
            {"session_id": session_id}
        ).sort("created_at", 1))
        
        # Convert ObjectIds to strings
        for message in messages:
            message["_id"] = str(message["_id"])
            if isinstance(message["session_id"], ObjectId):
                message["session_id"] = str(message["session_id"])
            if isinstance(message["user_id"], ObjectId):
                message["user_id"] = str(message["user_id"])
        
        return [Message(**message) for message in messages]
    
    @staticmethod
    async def get_message(message_id: str, user_id: str) -> Optional[Message]:
        """Get a specific message by ID"""
        # Get the message
        message = messages_collection.find_one({"_id": message_id})
        
        if not message:
            return None
        
        # Check if the message belongs to a session owned by the user
        session = sessions_collection.find_one({
            "_id": message["session_id"],
            "user_id": user_id
        })
        
        if not session:
            return None
        
        # Convert ObjectIds to strings
        message["_id"] = str(message["_id"])
        if isinstance(message["session_id"], ObjectId):
            message["session_id"] = str(message["session_id"])
        if isinstance(message["user_id"], ObjectId):
            message["user_id"] = str(message["user_id"])
        
        return Message(**message)
    
    @staticmethod
    async def get_session_context(session_id: str, user_id: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Get relevant context for a session"""
        # Check if the session exists and belongs to the user
        session = sessions_collection.find_one({
            "_id": session_id,
            "user_id": user_id
        })
        
        if not session or not session.get("vector_store_id"):
            return []
        
        # Search for relevant context
        documents = vector_store_manager.search_context(
            session["vector_store_id"],
            str(session["_id"]),
            query,
            k=k
        )
        
        # Convert to a list of dictionaries
        context = []
        for doc in documents:
            context.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return context 