import { useState, useRef, useEffect } from 'react';
import { Database, MessageSquare, Settings, Wifi, WifiOff } from 'lucide-react';
import Message from './Message';
import SqlResult from './SqlResult';
import SessionManager from './SessionManager';
import { executeQuery, getSessionInfo, getPaginatedResults } from '../lib/api';

// PaginationInfo interface to match the API response
interface PaginationInfo {
  table_id: string;
  current_page: number;
  total_pages: number;
  total_rows: number;
  page_size: number;
  has_next?: boolean;
  has_prev?: boolean;
}

interface TableInfo {
  name: string;
  description: string;
  sql: string;
  results: any[];
  row_count: number;
  table_id?: string;
  pagination?: PaginationInfo;
}

interface ChatMessage {
  id: string;
  isUser: boolean;
  text: string;
  timestamp: Date;
  query_type?: 'conversational' | 'sql' | 'analysis';
  sqlResult?: {
    sql: string;
    data?: any[];
    error?: string;
    pagination?: PaginationInfo;
    table_id?: string;
  };
  analysisResult?: {
    tables: TableInfo[];
    analysis_type: 'causal' | 'comparative';
  };
}

// Add a new interface for tracking pagination state
interface PaginationState {
  messageId: string;
  tableId: string;
  currentPage: number;
}

export default function ChatBot() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: 'welcome',
      isUser: false,
      text: 'Hello! I can help you query your database using natural language. How can I help you today?',
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionInfo, setSessionInfo] = useState<any>(null);
  const [showSessionManager, setShowSessionManager] = useState(false);
  const [paginationState, setPaginationState] = useState<PaginationState | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  useEffect(() => {
    inputRef.current?.focus();
    
    if (sessionId) {
      fetchSessionInfo();
    }
  }, [sessionId]);

  const fetchSessionInfo = async () => {
    if (!sessionId) return;
    
    try {
      const info = await getSessionInfo(sessionId);
      setSessionInfo(info);
    } catch (error) {
      console.error('Error fetching session info:', error);
      setSessionId(null);
      setSessionInfo(null);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!input.trim()) return;
    
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      isUser: true,
      text: input,
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);
    
    try {
      const result = await executeQuery(input, sessionId || undefined);
      
      let responseMessage = result.text || result.message || 'Query executed successfully.';
      const botMessage: ChatMessage = {
        id: `response-${Date.now()}`,
        isUser: false,
        text: responseMessage,
        timestamp: new Date(),
        query_type: result.query_type,
      };
      
      // Handle different response types
      if (result.query_type === 'sql') {
        botMessage.sqlResult = {
          sql: result.sql || '',
          data: result.data,
          error: result.error,
          pagination: result.pagination,
          table_id: result.table_id,
        };
        
        // Log for debugging
        console.log('SQL Result with pagination:', {
          tableId: result.table_id,
          pagination: result.pagination
        });
      } else if (result.query_type === 'analysis') {
        // Each table should have its own table_id for pagination
        const tablesWithIds = result.tables.map((table: any) => {
          console.log('Analysis table:', table);
          return {
            ...table,
            table_id: table.table_id || `table-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`,
            pagination: table.pagination
          };
        });
        
        botMessage.analysisResult = {
          tables: tablesWithIds,
          analysis_type: result.analysis_type
        };
      }
      // For conversational queries, we just use the text
      
      setMessages((prev) => [...prev, botMessage]);
    } catch (error: any) {
      const errorMessage: ChatMessage = {
        id: `error-${Date.now()}`,
        isUser: false,
        text: `Error: ${error.message || 'Failed to execute query'}`,
        timestamp: new Date(),
      };
      
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      inputRef.current?.focus();
    }
  };

  const handleSessionCreated = (newSessionId: string) => {
    setSessionId(newSessionId);
    
    const systemMessage: ChatMessage = {
      id: `system-${Date.now()}`,
      isUser: false,
      text: `✅ Connected to database successfully. Session ID: ${newSessionId}`,
      timestamp: new Date(),
    };
    
    setMessages((prev) => [...prev, systemMessage]);
  };

  const handlePageChange = async (messageId: string, tableId: string, newPage: number) => {
    if (!sessionId || !tableId) {
      console.error('Missing session ID or table ID for pagination', { sessionId, tableId });
      return;
    }
    
    console.log(`Fetching page ${newPage} for table ${tableId} in session ${sessionId}`);
    setIsProcessing(true);
    
    try {
      const result = await getPaginatedResults(sessionId, tableId, newPage);
      console.log('Paginated results:', result);
      
      // Make sure we have the current table_id (it might have changed in the response)
      const currentTableId = result.pagination?.table_id || tableId;
      
      // Update the message with the new data
      setMessages((prevMessages) => 
        prevMessages.map((msg) => {
          if (msg.id === messageId && msg.sqlResult) {
            return {
              ...msg,
              sqlResult: {
                ...msg.sqlResult,
                data: result.data,
                pagination: result.pagination,
                table_id: currentTableId
              }
            };
          } else if (msg.id === messageId && msg.analysisResult) {
            // For analysis results, find and update the specific table
            const updatedTables = msg.analysisResult.tables.map(table => {
              if (table.table_id === tableId) {
                return {
                  ...table,
                  results: result.data,
                  pagination: result.pagination,
                  table_id: currentTableId
                };
              }
              return table;
            });
            
            return {
              ...msg,
              analysisResult: {
                ...msg.analysisResult,
                tables: updatedTables,
              }
            };
          }
          return msg;
        })
      );
      
      // Update pagination state
      setPaginationState({
        messageId,
        tableId: currentTableId,
        currentPage: newPage,
      });
    } catch (error) {
      console.error('Error fetching paginated results:', error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-md shadow-lg border-b border-white/20 py-4 px-6 transition-all duration-300">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl shadow-lg">
              <MessageSquare className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
                NLP to SQL Assistant
              </h1>
              <p className="text-sm text-gray-500">Powered by AI • Natural Language Database Queries</p>
            </div>
          </div>
          
          <div className="flex items-center gap-4">
            {sessionId ? (
              <div className="flex items-center bg-green-50 px-4 py-2 rounded-full border border-green-200 transition-all duration-300 hover:bg-green-100">
                <Wifi className="h-4 w-4 text-green-600 mr-2" />
                <span className="text-sm font-medium text-green-700">
                  {sessionInfo?.db_info?.db_name || 'Connected'}
                </span>
                <div className="h-2 w-2 rounded-full bg-green-500 ml-2 animate-pulse"></div>
              </div>
            ) : (
              <div className="flex items-center bg-gray-50 px-4 py-2 rounded-full border border-gray-200">
                <WifiOff className="h-4 w-4 text-gray-400 mr-2" />
                <span className="text-sm text-gray-500">Not connected</span>
              </div>
            )}
            
            <button
              onClick={() => setShowSessionManager(true)}
              className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white text-sm py-2.5 px-4 rounded-xl shadow-lg transition-all duration-300 transform hover:scale-105 hover:shadow-xl"
            >
              <Database className="h-4 w-4" />
              <span>{sessionId ? 'Change Connection' : 'Connect Database'}</span>
            </button>
          </div>
        </div>
      </header>
      
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 scroll-smooth">
        <div className="max-w-5xl mx-auto">
          <div className="space-y-6">
            {messages.map((message, index) => (
              <div 
                key={message.id} 
                className="animate-in slide-in-from-bottom-2 fade-in duration-500"
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <Message
                  isUser={message.isUser}
                  content={message.text}
                  timestamp={message.timestamp}
                  isConversational={!message.isUser && message.query_type === 'conversational'}
                />
                {/* Render different types of results based on query_type */}
                {message.query_type === 'sql' && message.sqlResult && (
                  <div className="ml-4 mt-4 animate-in fade-in duration-700">
                    <SqlResult
                      sql={message.sqlResult.sql}
                      data={message.sqlResult.data}
                      error={message.sqlResult.error}
                      pagination={message.sqlResult.pagination}
                      onPageChange={(page) => handlePageChange(message.id, message.sqlResult?.table_id || '', page)}
                      sessionId={sessionId || undefined}
                      tableId={message.sqlResult.table_id}
                    />
                  </div>
                )}
                {message.query_type === 'analysis' && message.analysisResult && (
                  <div className="ml-4 mt-4 animate-in fade-in duration-700">
                    {message.analysisResult.tables.map((table, tableIndex) => (
                      <div key={tableIndex} className="mb-8 last:mb-0">
                        <SqlResult
                          sql={table.sql}
                          data={table.results}
                          title={table.name}
                          description={table.description}
                          pagination={table.pagination}
                          onPageChange={(page) => handlePageChange(message.id, table.table_id || '', page)}
                          sessionId={sessionId || undefined}
                          tableId={table.table_id}
                        />
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
          <div ref={messagesEndRef} />
          
          {isProcessing && (
            <div className="flex items-center justify-center py-8 animate-in fade-in duration-300">
              <div className="flex items-center space-x-3 bg-white/80 backdrop-blur-sm px-6 py-3 rounded-full shadow-lg border border-white/20">
                <div className="flex space-x-1">
                  <div className="h-2 w-2 rounded-full bg-blue-500 animate-bounce"></div>
                  <div className="h-2 w-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="h-2 w-2 rounded-full bg-blue-500 animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
                <span className="text-sm font-medium text-gray-600">Processing your query...</span>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Input Area */}
      <div className="bg-white/80 backdrop-blur-md border-t border-white/20 p-4 md:p-6 shadow-2xl">
        <div className="max-w-5xl mx-auto">
          <form onSubmit={handleSendMessage} className="relative">
            <div className="flex items-center bg-white rounded-2xl shadow-xl border border-gray-200/50 overflow-hidden transition-all duration-300 hover:shadow-2xl focus-within:shadow-2xl focus-within:border-blue-300">
              <input
                ref={inputRef}
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask me anything about your data..."
                className="flex-1 py-4 px-6 text-gray-800 placeholder-gray-400 bg-transparent focus:outline-none text-base"
                disabled={isProcessing}
              />
              <button
                type="submit"
                className={`m-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white p-3 rounded-xl transition-all duration-300 transform hover:scale-105 ${
                  isProcessing ? 'opacity-50 cursor-not-allowed scale-100' : 'hover:shadow-lg'
                }`}
                disabled={isProcessing || !input.trim()}
              >
                <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </button>
            </div>
          </form>
          
          {!sessionId && (
            <p className="text-center text-sm text-gray-500 mt-3 animate-pulse">
              💡 Connect to a database for persistent context and better results
            </p>
          )}
        </div>
      </div>
      
      {/* Session Manager Dialog */}
      <SessionManager
        isOpen={showSessionManager}
        onClose={() => setShowSessionManager(false)}
        onSessionCreated={handleSessionCreated}
      />
    </div>
  );
}