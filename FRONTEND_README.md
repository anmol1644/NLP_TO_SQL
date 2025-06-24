# NLP to SQL Frontend Integration Guide

This guide explains how to integrate the NLP to SQL backend with a Next.js frontend application.

## Overview

The NLP to SQL system provides a RESTful API that allows users to:
- Register and authenticate
- Create and manage workspaces connected to different databases
- Create chat sessions within workspaces
- Send natural language queries and receive SQL and results

## Setup

### Prerequisites

- Node.js 16+
- Next.js 13+
- React 18+

### Environment Variables

Create a `.env.local` file in your Next.js project with:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Authentication Integration

#### User Registration

```typescript
// components/RegisterForm.tsx
import { useState } from 'react';
import { useRouter } from 'next/router';

export default function RegisterForm() {
  const [formData, setFormData] = useState({
    email: '',
    username: '',
    full_name: '',
    password: ''
  });
  const [error, setError] = useState('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/register`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Registration failed');
      }
      
      // Redirect to login page
      router.push('/login');
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Form implementation...
}
```

#### User Login

```typescript
// components/LoginForm.tsx
import { useState } from 'react';
import { useRouter } from 'next/router';

export default function LoginForm() {
  const [formData, setFormData] = useState({
    username: '',
    password: ''
  });
  const [error, setError] = useState('');
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    try {
      // Convert form data to x-www-form-urlencoded format
      const formBody = new URLSearchParams();
      formBody.append('username', formData.username);
      formBody.append('password', formData.password);
      
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: formBody.toString(),
      });
      
      if (!response.ok) {
        throw new Error('Login failed');
      }
      
      const data = await response.json();
      
      // Store token in localStorage
      localStorage.setItem('auth_token', data.access_token);
      localStorage.setItem('token_expiry', data.expires_at);
      
      // Redirect to dashboard
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.message);
    }
  };

  // Form implementation...
}
```

### Auth Context

Create an authentication context to manage user state:

```typescript
// contexts/AuthContext.tsx
import { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: string;
  email: string;
  username: string;
  full_name: string;
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (token: string, expiry: string) => void;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in
    const token = localStorage.getItem('auth_token');
    if (token) {
      fetchUser(token);
    } else {
      setLoading(false);
    }
  }, []);

  const fetchUser = async (token: string) => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (response.ok) {
        const userData = await response.json();
        setUser(userData);
      } else {
        // Token invalid, logout
        logout();
      }
    } catch (error) {
      console.error('Failed to fetch user:', error);
      logout();
    } finally {
      setLoading(false);
    }
  };

  const login = (token: string, expiry: string) => {
    localStorage.setItem('auth_token', token);
    localStorage.setItem('token_expiry', expiry);
    fetchUser(token);
  };

  const logout = () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('token_expiry');
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ 
      user, 
      loading, 
      login, 
      logout,
      isAuthenticated: !!user
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}
```

### API Client

Create a reusable API client with authentication:

```typescript
// utils/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL;

async function fetchWithAuth(endpoint: string, options: RequestInit = {}) {
  const token = localStorage.getItem('auth_token');
  
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers,
  });
  
  if (response.status === 401) {
    // Unauthorized, clear token
    localStorage.removeItem('auth_token');
    localStorage.removeItem('token_expiry');
    window.location.href = '/login';
    throw new Error('Session expired. Please login again.');
  }
  
  return response;
}

export const api = {
  get: (endpoint: string) => fetchWithAuth(endpoint),
  
  post: (endpoint: string, data: any) => fetchWithAuth(endpoint, {
    method: 'POST',
    body: JSON.stringify(data),
  }),
  
  put: (endpoint: string, data: any) => fetchWithAuth(endpoint, {
    method: 'PUT',
    body: JSON.stringify(data),
  }),
  
  delete: (endpoint: string) => fetchWithAuth(endpoint, {
    method: 'DELETE',
  }),
};
```

### Workspace Management

```typescript
// components/WorkspaceManager.tsx
import { useState, useEffect } from 'react';
import { api } from '../utils/api';

export default function WorkspaceManager() {
  const [workspaces, setWorkspaces] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    fetchWorkspaces();
  }, []);
  
  const fetchWorkspaces = async () => {
    try {
      const response = await api.get('/workspaces');
      if (response.ok) {
        const data = await response.json();
        setWorkspaces(data);
      }
    } catch (error) {
      console.error('Failed to fetch workspaces:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const createWorkspace = async (workspaceData) => {
    try {
      const response = await api.post('/workspaces', workspaceData);
      if (response.ok) {
        const newWorkspace = await response.json();
        setWorkspaces([...workspaces, newWorkspace]);
        return newWorkspace;
      }
    } catch (error) {
      console.error('Failed to create workspace:', error);
      throw error;
    }
  };
  
  // Implementation for other workspace operations...
}
```

### Session Management

```typescript
// components/SessionManager.tsx
import { useState, useEffect } from 'react';
import { api } from '../utils/api';

export default function SessionManager({ workspaceId }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  
  useEffect(() => {
    if (workspaceId) {
      fetchSessions();
    }
  }, [workspaceId]);
  
  const fetchSessions = async () => {
    try {
      const response = await api.get(`/workspaces/${workspaceId}/sessions`);
      if (response.ok) {
        const data = await response.json();
        setSessions(data);
      }
    } catch (error) {
      console.error('Failed to fetch sessions:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const createSession = async (sessionData) => {
    try {
      const response = await api.post(`/workspaces/${workspaceId}/sessions`, sessionData);
      if (response.ok) {
        const newSession = await response.json();
        setSessions([...sessions, newSession]);
        return newSession;
      }
    } catch (error) {
      console.error('Failed to create session:', error);
      throw error;
    }
  };
  
  // Implementation for other session operations...
}
```

### Chat Interface

```typescript
// components/ChatBot.tsx
import { useState, useEffect, useRef } from 'react';
import { api } from '../utils/api';

export default function ChatBot({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [inputText, setInputText] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  const messagesEndRef = useRef(null);
  
  useEffect(() => {
    if (sessionId) {
      fetchMessages();
    }
  }, [sessionId]);
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  const fetchMessages = async () => {
    try {
      const response = await api.get(`/sessions/${sessionId}/messages`);
      if (response.ok) {
        const data = await response.json();
        setMessages(data);
      }
    } catch (error) {
      console.error('Failed to fetch messages:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const sendMessage = async (e) => {
    e.preventDefault();
    if (!inputText.trim() || isProcessing) return;
    
    const question = inputText.trim();
    setInputText('');
    setIsProcessing(true);
    
    // Add user message to UI immediately
    const userMessage = {
      content: question,
      role: 'user',
      created_at: new Date().toISOString()
    };
    setMessages([...messages, userMessage]);
    
    try {
      const response = await api.post(`/sessions/${sessionId}/query`, {
        question,
        auto_fix: true,
        max_attempts: 2
      });
      
      if (response.ok) {
        const data = await response.json();
        
        // Add assistant message from response
        if (data.assistant_message) {
          setMessages(prev => [...prev, data.assistant_message]);
        }
        
        // Handle query result
        handleQueryResult(data);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      
      // Add error message
      const errorMessage = {
        content: 'Sorry, there was an error processing your request.',
        role: 'assistant',
        created_at: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
    }
  };
  
  const handleQueryResult = (result) => {
    // Handle different types of results (SQL, conversational, analysis)
    // Implementation depends on your UI requirements
  };
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  // Chat UI implementation...
}
```

### Pagination Component

```typescript
// components/ResultsPagination.tsx
import { useState, useEffect } from 'react';
import { api } from '../utils/api';

export default function ResultsPagination({ sessionId, tableId, initialData }) {
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [totalPages, setTotalPages] = useState(1);
  const [results, setResults] = useState(initialData?.results || []);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    if (initialData) {
      setResults(initialData.results);
      if (initialData.pagination) {
        setTotalPages(initialData.pagination.total_pages);
      }
    }
  }, [initialData]);
  
  const fetchPage = async (page) => {
    setLoading(true);
    try {
      const response = await api.get(
        `/sessions/${sessionId}/results/${tableId}?page=${page}&page_size=${pageSize}`
      );
      
      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
        if (data.pagination) {
          setTotalPages(data.pagination.total_pages);
        }
        setCurrentPage(page);
      }
    } catch (error) {
      console.error('Failed to fetch page:', error);
    } finally {
      setLoading(false);
    }
  };
  
  // Pagination UI implementation...
}
```

## Protecting Routes

Create a higher-order component to protect authenticated routes:

```typescript
// components/ProtectedRoute.tsx
import { useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';

export default function ProtectedRoute({ children }) {
  const { isAuthenticated, loading } = useAuth();
  const router = useRouter();
  
  useEffect(() => {
    if (!loading && !isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, loading, router]);
  
  if (loading) {
    return <div>Loading...</div>;
  }
  
  return isAuthenticated ? children : null;
}
```

## Complete App Setup

Wrap your application with the auth provider:

```typescript
// pages/_app.tsx
import { AuthProvider } from '../contexts/AuthContext';

function MyApp({ Component, pageProps }) {
  return (
    <AuthProvider>
      <Component {...pageProps} />
    </AuthProvider>
  );
}

export default MyApp;
```

## Deployment Considerations

1. Update the `NEXT_PUBLIC_API_URL` to point to your production API endpoint
2. Implement proper error handling and loading states
3. Consider adding a token refresh mechanism
4. Add proper form validation
5. Implement proper UI components for displaying SQL results and tables

## Security Best Practices

1. Never store sensitive information in localStorage (JWT tokens are acceptable, but consider using HTTP-only cookies in production)
2. Implement proper CSRF protection
3. Use HTTPS in production
4. Validate all user inputs
5. Implement rate limiting on the frontend
6. Consider adding two-factor authentication for added security 