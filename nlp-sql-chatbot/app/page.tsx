'use client';

import dynamic from 'next/dynamic';

// Dynamically import the ChatBot component with no SSR to avoid hydration issues
const ChatBot = dynamic(() => import('../components/ChatBot'), { ssr: false });

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      <ChatBot />
    </main>
  );
} 