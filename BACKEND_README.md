# NLP to SQL Backend

This is the backend service for the NLP to SQL application, which allows users to interact with databases using natural language.

## Features

- **User Authentication**: JWT-based authentication system
- **Multi-workspace Support**: Users can create multiple workspaces, each connected to a different database
- **Session Management**: Each workspace can have multiple chat sessions
- **Vector Database Integration**: Conversation context is stored in a vector database for better context retrieval
- **Pagination**: Results are paginated for better performance
- **MongoDB Integration**: User data, workspaces, and sessions are stored in MongoDB

## Setup

### Prerequisites

- Python 3.8+
- MongoDB 4.4+
- PostgreSQL (for the target databases)

### Environment Variables

Create a `.env` file in the root directory with the following variables:

```
# MongoDB Connection
# For local MongoDB:
MONGO_URI=mongodb://localhost:27017
# For MongoDB Atlas:
# MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
MONGO_DB_NAME=nlp_to_sql_db
MONGO_CONNECT_TIMEOUT=5000

# JWT Authentication
SECRET_KEY=your-secret-key-for-jwt
ACCESS_TOKEN_EXPIRE_MINUTES=1440  # 24 hours

# Google API for embeddings
GOOGLE_API_KEY=your-google-api-key

# Vector Store
VECTOR_STORE_DIR=./vector_stores
```

#### MongoDB Connection String Format

- **Local MongoDB**: `mongodb://localhost:27017`
- **MongoDB Atlas**: `mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority`
- **MongoDB with Authentication**: `mongodb://<username>:<password>@localhost:27017`

Make sure to replace `<username>`, `<password>`, and `<cluster>` with your actual MongoDB credentials.

### Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the server:

```bash
uvicorn api:app --reload
```

## API Documentation

### Authentication

#### Register a new user
```
POST /auth/register
```
Request body:
```json
{
  "email": "user@example.com",
  "username": "username",
  "full_name": "Full Name",
  "password": "password"
}
```

#### Login
```
POST /auth/token
```
Form data:
- `username`: User's email
- `password`: User's password

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_at": "2023-05-30T12:00:00.000Z"
}
```

#### Get current user
```
GET /auth/me
```
Headers:
- `Authorization`: Bearer {access_token}

### Workspaces

#### Create a workspace
```
POST /workspaces
```
Request body:
```json
{
  "name": "My Workspace",
  "description": "Description of the workspace",
  "db_connection": {
    "db_name": "postgres",
    "username": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432",
    "db_type": "postgresql"
  }
}
```

#### Get all workspaces
```
GET /workspaces
```

#### Get a workspace
```
GET /workspaces/{workspace_id}
```

#### Update a workspace
```
PUT /workspaces/{workspace_id}
```

#### Delete a workspace
```
DELETE /workspaces/{workspace_id}
```

### Sessions

#### Create a session
```
POST /workspaces/{workspace_id}/sessions
```
Request body:
```json
{
  "name": "My Session",
  "description": "Description of the session"
}
```

#### Get all sessions for a workspace
```
GET /workspaces/{workspace_id}/sessions
```

#### Get a session
```
GET /sessions/{session_id}
```

#### Delete a session
```
DELETE /sessions/{session_id}
```

### Messages and Queries

#### Get all messages for a session
```
GET /sessions/{session_id}/messages
```

#### Send a query
```
POST /sessions/{session_id}/query
```
Request body:
```json
{
  "question": "Show me all sales from 2014",
  "auto_fix": true,
  "max_attempts": 2
}
```

#### Get paginated results
```
GET /sessions/{session_id}/results/{table_id}?page=1&page_size=10
```

## Database Schema

The application uses MongoDB with the following collections:

### Users Collection
```json
{
  "_id": ObjectId,
  "email": "user@example.com",
  "username": "username",
  "full_name": "Full Name",
  "hashed_password": "hashed_password",
  "is_active": true,
  "created_at": ISODate,
  "last_login": ISODate
}
```

### Workspaces Collection
```json
{
  "_id": ObjectId,
  "name": "Workspace Name",
  "description": "Workspace Description",
  "user_id": ObjectId,
  "db_connection": {
    "db_name": "postgres",
    "username": "postgres",
    "password": "password",
    "host": "localhost",
    "port": "5432",
    "db_type": "postgresql"
  },
  "created_at": ISODate,
  "updated_at": ISODate,
  "is_active": true
}
```

### Sessions Collection
```json
{
  "_id": ObjectId,
  "name": "Session Name",
  "description": "Session Description",
  "workspace_id": ObjectId,
  "user_id": ObjectId,
  "vector_store_id": "uuid",
  "created_at": ISODate,
  "updated_at": ISODate,
  "last_active": ISODate,
  "is_active": true
}
```

### Messages Collection
```json
{
  "_id": ObjectId,
  "session_id": ObjectId,
  "content": "Message content",
  "role": "user|assistant",
  "created_at": ISODate,
  "metadata": {}
}
```

## Vector Database

The application uses Chroma DB with Google's Gemini embeddings to store conversation context. Each session has its own vector store, which is used to retrieve relevant context for future queries.

## Troubleshooting

### MongoDB Connection Issues

If you encounter MongoDB connection issues:

1. **Check your connection string**: Make sure the format is correct for your MongoDB setup
2. **Network access**: Ensure your IP is allowed in MongoDB Atlas network access settings
3. **Credentials**: Verify username and password are correct
4. **Timeouts**: You can increase `MONGO_CONNECT_TIMEOUT` if needed

### Fallback Mode

The application includes a fallback mode that uses in-memory storage if MongoDB is unavailable. This is suitable for development and testing but not for production use.

## Security Considerations

- Database passwords are stored in MongoDB. In a production environment, consider encrypting these passwords.
- JWT tokens have a 24-hour expiration by default. Adjust this as needed for your security requirements.
- The API uses CORS with "*" as the allowed origin. In production, specify your frontend domains.

## Dependencies

- FastAPI: Web framework
- PyMongo: MongoDB client
- PyJWT: JWT authentication
- Passlib: Password hashing
- LangChain: Vector store integration
- Google Generative AI: Embeddings and LLM integration
- Chroma DB: Vector database 