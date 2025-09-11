// src/components/TrueLensChat.jsx - INTEGRATE WITH YOUR EXISTING CHAT
import React, { useState, useRef, useEffect } from "react";
import { Send, Sparkles, Loader2 } from "lucide-react";
import { useImageSearch } from "../hooks/useImageSearch";
import ImageResults from "./imageResults";

const TrueLensChat = () => {
  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const { messages, sendMessage, clearChat, isLoading } = useImageSearch();

  console.log("Messages:", messages);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const messageText = inputMessage;
    setInputMessage("");

    try {
      await sendMessage(messageText);
    } catch (error) {
      console.error("Failed to send message:", error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    setInputMessage(suggestion);
  };

  const formatMessage = (content: string) => {
    return content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Header */}
      <div className="bg-white shadow-sm border-b px-6 py-4">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <div className="flex items-center">
            <Sparkles className="h-8 w-8 text-indigo-600 mr-3" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">TrueLensAI</h1>
              <p className="text-gray-600">AI-Powered Art Discovery</p>
            </div>
          </div>
          <button
            onClick={clearChat}
            className="bg-gray-100 hover:bg-gray-200 text-gray-700 px-4 py-2 rounded-lg text-sm transition-colors"
          >
            New Search
          </button>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4">
        <div className="max-w-6xl mx-auto space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.type === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`max-w-4xl ${
                  message.type === "user"
                    ? "bg-indigo-600 text-white"
                    : "bg-white text-gray-800 shadow-md"
                } rounded-2xl p-6 border`}
              >
                <div
                  className="whitespace-pre-wrap leading-relaxed"
                  dangerouslySetInnerHTML={{
                    __html: formatMessage(message.content),
                  }}
                />

                {/* Image Results */}
                {message.images && <ImageResults images={message.images} />}

                {/* Suggestions */}
                {message.suggestions && message.suggestions.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-sm font-medium text-gray-700 mb-2">
                      💡 Try these examples:
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {message.suggestions
                        .slice(0, 3)
                        .map((suggestion, idx) => (
                          <button
                            key={idx}
                            onClick={() => handleSuggestionClick(suggestion)}
                            className="text-xs bg-gray-100 hover:bg-gray-200 text-gray-700 px-3 py-1 rounded-full transition-colors"
                          >
                            {suggestion}
                          </button>
                        ))}
                    </div>
                  </div>
                )}

                <div className="text-xs mt-3 opacity-70 flex items-center justify-between">
                  <span>{message.timestamp.toLocaleTimeString()}</span>
                  {message.processingTime && (
                    <span>⚡ {message.processingTime.toFixed(1)}s</span>
                  )}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white text-gray-800 rounded-2xl p-6 shadow-md flex items-center">
                <Loader2 className="w-5 h-5 mr-3 animate-spin text-indigo-600" />
                <span>TrueLensAI is analyzing artwork...</span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input */}
      <div className="bg-white border-t px-6 py-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex space-x-4">
            <div className="flex-1">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Describe the artwork you're looking for... (e.g., 'I need a realistic mountain logo for my hiking company')"
                className="w-full p-4 border border-gray-300 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent resize-none shadow-sm"
                rows={2}
                disabled={isLoading}
              />
            </div>
            <button
              onClick={handleSendMessage}
              disabled={isLoading || !inputMessage.trim()}
              className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white p-4 rounded-2xl transition-colors flex items-center justify-center min-w-[60px] shadow-sm"
            >
              {isLoading ? (
                <Loader2 className="w-6 h-6 animate-spin" />
              ) : (
                <Send className="w-6 h-6" />
              )}
            </button>
          </div>

          {/* Quick Actions */}
          <div className="mt-3 flex flex-wrap gap-2">
            {[
              "I need a minimalist logo for my tech startup",
              "Looking for abstract art to decorate my modern living room",
              "Want a realistic nature poster for my bedroom",
              "Need vintage-style artwork for my coffee shop",
            ].map((suggestion, index) => (
              <button
                key={index}
                onClick={() => handleSuggestionClick(suggestion)}
                className="text-sm bg-indigo-50 hover:bg-indigo-100 text-indigo-700 px-4 py-2 rounded-full transition-colors border border-indigo-200"
                disabled={isLoading}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrueLensChat;
