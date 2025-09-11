import React, { useEffect } from "react";
import type { ArtResult, Message } from "../../types";
import Footer from "../ui/Footer";
import ArtCard from "../ui/ArtCard";
import LoaderIcon from "../ui/LoaderIcon";
import ImageResults from "../imageResults";

interface InputPageProps {
  inputValue: string;
  onInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSearch: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onGoBack: () => void;
  isLoading: boolean;
  messages: Message[];
  currentTime: string;
  searchInputRef: React.RefObject<HTMLInputElement | null>;
  isFadingOut: boolean;
}

const InputPage: React.FC<InputPageProps> = ({
  inputValue,
  onInputChange,
  onSearch,
  onGoBack,
  isLoading,
  messages,
  currentTime,
  searchInputRef,
  isFadingOut,
}) => {
  const hasSearched = isLoading || messages.length > 1; // More than just the initial message
  const latestAssistantMessage = messages.find(
    (m) => m.type === "assistant" && m.images && m.images.length > 0
  );

  // 👇 Auto-focus and insert key when user types anywhere
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don’t steal focus if typing in another input/textarea/select
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }

      if (searchInputRef.current) {
        // Focus the input
        searchInputRef.current.focus();

        // If it’s a printable key (letters, numbers, symbols)
        if (e.key.length === 1) {
          // Replace current value with the typed char
          searchInputRef.current.value = e.key;

          // Fire an input event so React state updates
          const inputEvent = new Event("input", { bubbles: true });
          searchInputRef.current.dispatchEvent(inputEvent);
        }

        e.preventDefault();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [searchInputRef]);

  return (
    <>
      {/* Custom styles for animations */}
      <style>{`
                @keyframes fadeInUp {
                    from {
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                .animate-fade-in-up {
                    animation: fadeInUp 0.6s ease-out forwards;
                }
            `}</style>

      <div
        className={`main-container flex flex-col p-4 sm:p-6 md:p-8 relative transition-opacity duration-500 ${
          isFadingOut ? "opacity-0" : "opacity-100"
        }`}
      >
        <header className="w-full flex justify-start items-center z-20">
          <button
            onClick={onGoBack}
            className="w-10 h-10 bg-stone-800/50 rounded-full flex items-center justify-center cursor-pointer transition-colors hover:bg-stone-700/80"
            aria-label="Go back"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="#f5f5f4"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="19" y1="12" x2="5" y2="12"></line>
              <polyline points="12 19 5 12 12 5"></polyline>
            </svg>
          </button>
        </header>

        <main
          className={`flex-grow flex flex-col items-center w-full max-w-6xl mx-auto transition-all duration-700 ease-in-out ${
            hasSearched ? "justify-start pt-12 sm:pt-16" : "justify-center"
          }`}
        >
          {/* Input Section */}
          <div className="w-full text-center">
            <input
              ref={searchInputRef}
              type="text"
              value={inputValue}
              onChange={onInputChange}
              onKeyDown={onSearch}
              placeholder="An abstract feeling, like nostalgia or wanderlust..."
              className={`w-full bg-transparent border-none text-stone-100 text-center focus:outline-none placeholder-stone-600 tracking-tight transition-all duration-700 ease-in-out
                                ${
                                  hasSearched
                                    ? "text-2xl md:text-3xl lg:text-4xl"
                                    : "text-4xl md:text-5xl lg:text-6xl"
                                }
                            `}
            />
            <p
              className={`text-center text-stone-500 text-sm mt-6 transition-opacity duration-500 delay-300 ${
                inputValue && !hasSearched ? "opacity-100" : "opacity-0"
              }`}
            >
              Press Enter to curate
            </p>
          </div>

          {/* Loading State */}
          {isLoading && (
            <div className="mt-20">
              <LoaderIcon />
            </div>
          )}

          {/* Image Results */}
          {latestAssistantMessage && latestAssistantMessage.images && (
            <div className="w-full mt-16 sm:mt-24">
              <ImageResults images={latestAssistantMessage.images} />
            </div>
          )}
        </main>
        <Footer currentTime={currentTime} />
      </div>
    </>
  );
};

export default InputPage;
