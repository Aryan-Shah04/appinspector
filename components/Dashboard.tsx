import React, { useState, useRef, useEffect } from 'react';
import { AppSearchResult, AppAnalysis, ChatMessage } from '../types';
import { chatWithApp } from '../services/geminiService';
import MarkdownText from './MarkdownText';
import { 
  ShieldCheck, 
  MessageSquare, 
  Info, 
  Send, 
  Bot, 
  ExternalLink, 
  Download,
  Star,
  ChevronLeft,
  Calendar,
  User
} from 'lucide-react';

interface DashboardProps {
  app: AppSearchResult;
  analysis: AppAnalysis;
  onBack: () => void;
}

const Dashboard: React.FC<DashboardProps> = ({ app, analysis, onBack }) => {
  const [activeTab, setActiveTab] = useState<'reviews' | 'authenticity' | 'background'>('reviews');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatHistory, isChatLoading]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isChatLoading) return;

    const userMsg: ChatMessage = { role: 'user', content: inputMessage };
    setChatHistory(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsChatLoading(true);

    try {
      const responseText = await chatWithApp(chatHistory, userMsg.content, app, analysis);
      const botMsg: ChatMessage = { role: 'model', content: responseText };
      setChatHistory(prev => [...prev, botMsg]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMsg: ChatMessage = { role: 'model', content: "Sorry, I encountered an error. Please try again." };
      setChatHistory(prev => [...prev, errorMsg]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const renderAnalysisContent = () => {
    switch (activeTab) {
      case 'reviews':
        return (
          <div className="space-y-4 animate-fade-in">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              <Star className="h-5 w-5 text-indigo-600 mr-2" />
              Review Analysis
            </h3>
            <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
              <MarkdownText content={analysis.reviewSummary} />
            </div>
          </div>
        );
      case 'authenticity':
        return (
          <div className="space-y-4 animate-fade-in">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              <ShieldCheck className="h-5 w-5 text-emerald-600 mr-2" />
              Authenticity & Safety
            </h3>
            <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
              <MarkdownText content={analysis.authenticity} />
            </div>
          </div>
        );
      case 'background':
        return (
          <div className="space-y-4 animate-fade-in">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              <Info className="h-5 w-5 text-blue-600 mr-2" />
              Developer Background
            </h3>
            <div className="bg-white p-6 rounded-2xl border border-gray-100 shadow-sm">
              <MarkdownText content={analysis.background} />
            </div>
          </div>
        );
    }
  };

  return (
    <div className="max-w-7xl mx-auto h-[calc(100vh-6rem)] flex flex-col md:flex-row gap-6 animate-fade-in">
      
      {/* Left Panel: Analysis */}
      <div className="w-full md:w-1/2 lg:w-3/5 flex flex-col overflow-hidden rounded-3xl bg-gray-50 border border-gray-200 shadow-xl">
        {/* Header */}
        <div className="bg-white border-b border-gray-100 p-6">
          <button 
            onClick={onBack}
            className="flex items-center text-sm text-gray-500 hover:text-indigo-600 mb-4 transition-colors w-fit"
          >
            <ChevronLeft className="h-4 w-4 mr-1" /> Back to Search
          </button>
          
          <div className="mb-4">
            <h1 className="text-3xl font-bold text-gray-900 mb-1">{app.name}</h1>
            <div className="flex items-center text-gray-500 font-medium">
               <User className="h-4 w-4 mr-1" />
               {app.developer}
            </div>
          </div>

          <div className="flex flex-wrap gap-3">
             <div className="flex-1 min-w-[100px] flex flex-col items-center justify-center p-3 bg-indigo-50 rounded-xl">
                <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wide mb-1">Downloads</span>
                <div className="flex items-center text-indigo-700 font-bold text-lg">
                   <Download className="h-4 w-4 mr-1" />
                   {analysis.downloads || 'N/A'}
                </div>
             </div>
             
             <div className="flex-1 min-w-[100px] flex flex-col items-center justify-center p-3 bg-yellow-50 rounded-xl">
                <span className="text-xs font-semibold text-yellow-500 uppercase tracking-wide mb-1">Rating</span>
                <div className="flex items-center text-yellow-700 font-bold text-lg">
                   <Star className="h-4 w-4 mr-1 fill-current" />
                   {analysis.rating || 'N/A'}
                </div>
             </div>

             {analysis.lastUpdated && (
               <div className="flex-1 min-w-[100px] flex flex-col items-center justify-center p-3 bg-green-50 rounded-xl">
                  <span className="text-xs font-semibold text-green-500 uppercase tracking-wide mb-1">Updated</span>
                  <div className="flex items-center text-green-700 font-bold text-sm text-center whitespace-nowrap">
                     <Calendar className="h-4 w-4 mr-1" />
                     {analysis.lastUpdated}
                  </div>
               </div>
             )}
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-200 bg-white px-6">
          {(['reviews', 'authenticity', 'background'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`pb-4 pt-4 px-4 font-medium text-sm transition-all border-b-2 ${
                activeTab === tab
                  ? 'border-indigo-600 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 bg-gray-50/50">
          {renderAnalysisContent()}

          {analysis.groundingUrls && analysis.groundingUrls.length > 0 && (
            <div className="mt-8 pt-6 border-t border-gray-200">
              <h4 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Sources</h4>
              <ul className="space-y-2">
                {analysis.groundingUrls.map((url, idx) => (
                  <li key={idx}>
                    <a 
                      href={url} 
                      target="_blank" 
                      rel="noopener noreferrer"
                      className="flex items-center text-xs text-gray-500 hover:text-indigo-600 transition-colors truncate"
                    >
                      <ExternalLink className="h-3 w-3 mr-2 flex-shrink-0" />
                      {url}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>

      {/* Right Panel: Chat */}
      <div className="w-full md:w-1/2 lg:w-2/5 flex flex-col bg-white rounded-3xl border border-gray-200 shadow-xl overflow-hidden">
        <div className="p-4 border-b border-gray-100 bg-indigo-600 text-white flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Bot className="h-5 w-5" />
            <h3 className="font-semibold">AI Assistant</h3>
          </div>
          <span className="text-xs bg-indigo-500 px-2 py-1 rounded-md opacity-90 truncate max-w-[150px]" title="gemini-2.5-flash-native-audio-dialog">
            gemini-2.5-flash-native-audio-dialog
          </span>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
          {chatHistory.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-gray-400 p-8 text-center">
              <MessageSquare className="h-12 w-12 mb-3 opacity-20" />
              <p className="text-sm">Ask me anything about {app.name}!</p>
              <p className="text-xs mt-2 opacity-60">I can explain reviews, safety concerns, or comparisons.</p>
            </div>
          )}
          
          {chatHistory.map((msg, idx) => (
            <div 
              key={idx} 
              className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div 
                className={`flex max-w-[85%] rounded-2xl p-4 text-sm ${
                  msg.role === 'user' 
                    ? 'bg-indigo-600 text-white rounded-tr-none' 
                    : 'bg-white text-gray-800 border border-gray-100 shadow-sm rounded-tl-none'
                }`}
              >
                {msg.role === 'model' && (
                  <Bot className="h-4 w-4 text-indigo-500 mr-2 flex-shrink-0 mt-0.5" />
                )}
                <div className="w-full">
                   <MarkdownText content={msg.content} className={msg.role === 'user' ? 'text-white' : 'text-gray-800'} />
                </div>
              </div>
            </div>
          ))}
          
          {isChatLoading && (
             <div className="flex justify-start w-full">
               <div className="bg-white border border-gray-100 shadow-sm rounded-2xl rounded-tl-none p-4 flex items-center space-x-2">
                 <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                 <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                 <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
               </div>
             </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="p-4 bg-white border-t border-gray-100">
          <div className="relative">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Type your question..."
              className="w-full pl-4 pr-12 py-3 bg-gray-50 border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all text-sm"
            />
            <button 
              type="submit"
              disabled={!inputMessage.trim() || isChatLoading}
              className="absolute right-2 top-1/2 transform -translate-y-1/2 p-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:bg-gray-400 transition-colors"
            >
              <Send className="h-4 w-4" />
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Dashboard;