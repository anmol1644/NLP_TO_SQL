import { User, Bot } from 'lucide-react';
import { formatRelative } from 'date-fns';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface MessageProps {
  isUser: boolean;
  content: string;
  timestamp: Date;
  isConversational?: boolean;
}

export default function Message({ isUser, content, timestamp, isConversational = false }: MessageProps) {
  const formattedTime = formatRelative(new Date(timestamp), new Date());
  
  // Fix table formatting issues by ensuring proper markdown syntax
  let processedContent = content;
  
  // Fix common table formatting issues
  if (content.includes('|')) {
    // Remove extra spaces between table rows that break markdown tables
    processedContent = content
      .replace(/\|\s*\n\s*\|/g, '|\n|')
      .replace(/\|\s*\n\n\s*\|/g, '|\n|');
    
    // Fix malformed table headers/separators
    const lines = processedContent.split('\n');
    for (let i = 0; i < lines.length; i++) {
      // If this looks like a table separator row with errors
      if (lines[i].trim().startsWith('|') && lines[i].includes('--') && !lines[i].includes('|--')) {
        // Fix the separator line format
        lines[i] = lines[i].replace(/\s*\|\s*/g, ' | ').replace(/\s*-+\s*/g, ' --- ');
      }
    }
    processedContent = lines.join('\n');
  }

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="flex-shrink-0">
          <div className="h-9 w-9 rounded-full shadow-sm bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
            <Bot className="h-5 w-5 text-white" />
          </div>
        </div>
      )}
      
      <div className={`max-w-[80%] sm:max-w-[70%] ${isUser ? 'order-1' : 'order-2'}`}>
        <div 
          className={`px-4 py-3 rounded-xl shadow-sm ${
            isUser
              ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white'
              : isConversational
                ? 'bg-gradient-to-r from-indigo-50 to-purple-50 border border-indigo-100'
                : 'bg-white border border-gray-100'
          }`}
        >
          <div 
            className={`prose prose-sm max-w-none ${
              isUser 
                ? 'prose-invert' 
                : isConversational
                  ? 'prose-indigo'
                  : 'prose-gray'
            } prose-table:table-auto prose-table:w-full prose-td:p-2 prose-th:p-2 prose-thead:bg-gray-100 prose-tr:border-b prose-tr:border-gray-200`}
          >
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{processedContent}</ReactMarkdown>
          </div>
        </div>
        <div 
          className={`text-xs text-gray-400 mt-1 ${
            isUser ? 'text-right' : 'text-left'
          }`}
        >
          {formattedTime}
        </div>
      </div>
      
      {isUser && (
        <div className="flex-shrink-0 order-3">
          <div className="h-9 w-9 rounded-full shadow-sm bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center">
            <User className="h-5 w-5 text-white" />
          </div>
        </div>
      )}
    </div>
  );
}