// src/hooks/useImageSearch.js - NEW FILE
import { useState, useCallback } from "react";
import type { Message } from "../types/index";

export const useImageSearch = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      type: "assistant",
      content:
        "Hi! I'm TrueLensAI 🎨 I help you find artwork that perfectly matches your vision using advanced AI technology. What kind of artwork are you looking for today?",
      timestamp: new Date(),
      images: null,
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);

  const sendMessage = useCallback(
    async (messageText: string, userId = "anonymous") => {
      if (!messageText.trim() || isLoading) return;

      // Add user message
      const userMessage: Message = {
        id: Date.now(),
        type: "user",
        content: messageText,
        timestamp: new Date(),
        images: null,
      };

      setMessages((prev) => [...prev, userMessage]);
      setIsLoading(true);

      try {
        console.log(import.meta.env.VITE_TARGET);
        const response = await fetch(
          `${import.meta.env.VITE_TARGET}/api/chat`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              message: messageText,
              session_id: sessionId,
              user_id: userId,
            }),
          }
        );

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // Update session ID if provided
        if (data.session_id && !sessionId) {
          setSessionId(data.session_id);
        }

        // Add assistant response
        const assistantMessage: Message = {
          id: Date.now() + 1,
          type: "assistant",
          content: data.response as string,
          timestamp: new Date(),
          images: (data.images as Message["images"]) || null,
          suggestions: (data.suggestions as string[]) || null,
          processingTime: data.processing_time as number,
        };

        setMessages((prev) => [...prev, assistantMessage]);
        return data;
      } catch (error) {
        console.error("Error sending message:", error);

        let errorContent =
          "Sorry, I encountered an error. Please make sure the backend server is running and try again. 🤖";

        // Handle specific error types
        if (error instanceof Error) {
          if (error.message.includes("HTTP error! status: 500")) {
            errorContent =
              "I'm experiencing a technical issue with the AI service. Please try again in a moment. 🔧";
          } else if (
            error.message.includes("parsing") ||
            error.message.includes("OUTPUT_PARSING_FAILURE")
          ) {
            errorContent =
              "I had trouble understanding the response. Please try rephrasing your request. 🤔";
          } else if (
            error.message.includes("timeout") ||
            error.message.includes("network")
          ) {
            errorContent =
              "The request is taking longer than expected. Please try again. ⏱️";
          }
        }

        const errorMessage: Message = {
          id: Date.now() + 1,
          type: "assistant",
          content: errorContent,
          timestamp: new Date(),
          images: null,
        };

        setMessages((prev) => [...prev, errorMessage]);
        throw error;
      } finally {
        setIsLoading(false);
      }
    },
    [sessionId, isLoading]
  );

  const clearChat = useCallback(() => {
    setMessages([
      {
        id: 1,
        type: "assistant",
        content:
          "Hi! I'm TrueLensAI 🎨 I help you find artwork that perfectly matches your vision using advanced AI technology. What kind of artwork are you looking for today?",
        timestamp: new Date(),
        images: null,
      },
    ]);
    setSessionId(null);
  }, []);

  return {
    messages,
    sendMessage,
    clearChat,
    isLoading,
    sessionId,
  };
};
