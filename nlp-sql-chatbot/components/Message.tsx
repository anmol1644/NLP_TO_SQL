import { ReactNode } from 'react';
import { Bot, User } from 'lucide-react';

interface MessageProps {
  isUser: boolean;
  content: ReactNode;
  timestamp?: Date;
}

export default function Message({ isUser, content, timestamp }: MessageProps) {
  return (
    <div className={`flex w-full mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-[85%] gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        {/* Avatar */}
        <div className={`flex-shrink-0 h-10 w-10 rounded-full flex items-center justify-center shadow-lg transition-all duration-300 ${
          isUser 
            ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white' 
            : 'bg-gradient-to-r from-emerald-500 to-teal-600 text-white'
        }`}>
          {isUser ? <User className="h-5 w-5" /> : <Bot className="h-5 w-5" />}
        </div>
        
        {/* Message Content */}
        <div className={`relative transition-all duration-300 hover:shadow-lg ${
          isUser ? 'transform hover:-translate-y-1' : 'transform hover:-translate-y-1'
        }`}>
          <div
            className={`rounded-2xl px-5 py-4 shadow-md transition-all duration-300 ${
              isUser
                ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-br-md'
                : 'bg-white text-gray-800 rounded-bl-md border border-gray-100 shadow-lg'
            }`}
          >
            {typeof content === 'string' ? (
              <p className="text-sm leading-relaxed font-medium">{content}</p>
            ) : (
              content
            )}
            
            {timestamp && (
              <div className={`text-xs mt-2 ${
                isUser ? 'text-blue-100' : 'text-gray-400'
              }`}>
                {timestamp.toLocaleTimeString([], { 
                  hour: '2-digit', 
                  minute: '2-digit',
                  hour12: true 
                })}
              </div>
            )}
          </div>
          
          {/* Message tail */}
          <div className={`absolute top-4 ${
            isUser 
              ? 'right-0 translate-x-full' 
              : 'left-0 -translate-x-full'
          }`}>
            <div className={`w-0 h-0 ${
              isUser
                ? 'border-t-[8px] border-t-transparent border-l-[12px] border-l-blue-600 border-b-[8px] border-b-transparent'
                : 'border-t-[8px] border-t-transparent border-r-[12px] border-r-white border-b-[8px] border-b-transparent'
            }`} />
          </div>
        </div>
      </div>
    </div>
  );
}