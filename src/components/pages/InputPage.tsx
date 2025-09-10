import React, { useEffect, useState, useRef } from 'react';

import type { ArtResult } from '../../types';
import Footer from '../ui/Footer';
import ArtCard from '../ui/ArtCard';
import LoaderIcon from '../ui/LoaderIcon';

// Define a type for a chat message as a structured object
type ChatMessage = {
  role: 'human' | 'ai';
  content: string;
};

interface InputPageProps {
  onGoBack: () => void;
  currentTime: string;
  isFadingOut: boolean;
}

const InputPage: React.FC<InputPageProps> = ({
  onGoBack,
  currentTime,
  isFadingOut,
}) => {
    const [currentQuestion, setCurrentQuestion] = useState('Connecting to the assistant...');
    const [answer, setAnswer] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [results, setResults] = useState<ArtResult[]>([]);
    
    // State will now hold an array of message objects
    const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);

    const searchInputRef = useRef<HTMLInputElement | null>(null);
    const hasSearchedOrIsLoading = isLoading || results.length > 0;

    const postMessage = async (message: string) => {
        setIsLoading(true);
        setCurrentQuestion('');

        try {
            const response = await fetch(`http://localhost:8000/invoke`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    input: message,
                    chat_history: chatHistory,
                })
            });

            if (!response.ok) {
                const errorBody = await response.text();
                console.error("API Error Response:", errorBody);
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            const output = data.output;
            
            const userMessage: ChatMessage = { role: 'human', content: message };

            if (Array.isArray(output)) {
                const formattedResults = output.map((url, index) => ({
                    title: `Result ${index + 1}`,
                    artist: new URL(url).hostname,
                    match: Math.floor(Math.random() * (99 - 90 + 1) + 90),
                    imageUrl: url
                }));
                setResults(formattedResults);
                setCurrentQuestion('');
            } else if (typeof output === 'string') {
                const aiMessage: ChatMessage = { role: 'ai', content: output };
                setCurrentQuestion(output);
                setChatHistory(prevHistory => [...prevHistory, userMessage, aiMessage]);
            }

        } catch (error) {
            console.error("Failed to fetch from agent:", error);
            setCurrentQuestion("Sorry, an error occurred. Please check the console.");
        } finally {
            setIsLoading(false);
            setAnswer('');
        }
    };
    
    useEffect(() => {
        (async () => {
            await postMessage("I want to find an image.");
        })();
    }, []);

    useEffect(() => {
        if (!isLoading) {
            searchInputRef.current?.focus();
        }
    }, [currentQuestion, isLoading]);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter' && answer.trim() !== '' && !isLoading) {
            postMessage(answer.trim());
        }
    };

    return (
        <div
            className={`main-container flex flex-col p-4 sm:p-6 md:p-8 relative min-h-screen transition-opacity duration-500 ${
                isFadingOut ? "opacity-0" : "opacity-100"
            }`}
        >
            <header className="w-full flex justify-start items-center z-20">
                <button 
                    onClick={onGoBack} 
                    className="w-10 h-10 bg-stone-800/50 rounded-full flex items-center justify-center cursor-pointer transition-colors hover:bg-stone-700/80"
                    aria-label="Go back"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24" fill="none" stroke="#f5f5f4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
                </button>
            </header>

            <main className={`flex-grow flex flex-col items-center w-full max-w-6xl mx-auto transition-all duration-700 ease-in-out ${hasSearchedOrIsLoading ? 'justify-start pt-12 sm:pt-16' : 'justify-center'}`}>
                {!hasSearchedOrIsLoading && !isLoading && (
                    <div className="w-full text-center">
                        <h2 className="font-brand text-4xl md:text-5xl lg:text-6xl text-stone-100 mb-8 transition-opacity duration-500">
                            {currentQuestion}
                        </h2>
                        <input
                            ref={searchInputRef}
                            type="text"
                            value={answer}
                            onChange={(e) => setAnswer(e.target.value)}
                            onKeyDown={handleKeyDown}
                            placeholder="Type your answer..."
                            className="w-full bg-transparent border-none text-stone-100 text-center focus:outline-none placeholder-stone-600 tracking-tight transition-all duration-700 ease-in-out text-4xl md:text-5xl lg:text-6xl"
                            disabled={isLoading}
                        />
                        <p className="text-center text-stone-500 text-sm mt-6 transition-opacity duration-500 delay-300">
                            Press Enter to continue
                        </p>
                    </div>
                )}

                {isLoading && (
                    <div>
                        <LoaderIcon />
                    </div>
                )}

                {results.length > 0 && !isLoading && (
                    <div className="w-full mt-16 sm:mt-24">
                        <div className="grid grid-cols-2 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6">
                            {results.map((item, index) => (
                                <div 
                                    key={(item as any).id || index} 
                                    className="animate-fade-in-up" 
                                    style={{ animationDelay: `${index * 100}ms`, opacity: 0 }}
                                >
                                    <ArtCard item={item} index={index} />
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </main>
            
            <Footer currentTime={currentTime} />

            <style>{`
                @keyframes fadeInUp {
                    from { opacity: 0; transform: translateY(20px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .animate-fade-in-up { animation: fadeInUp 0.6s ease-out forwards; }
            `}</style>
        </div>
    );
};

export default InputPage;