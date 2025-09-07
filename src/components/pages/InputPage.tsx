import React from "react";
import type { ArtResult } from "../../types";
import Footer from "../ui/Footer";
import ArtCard from "../ui/ArtCard";
import LoaderIcon from "../ui/LoaderIcon";

interface InputPageProps {
  inputValue: string;
  onInputChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onSearch: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onGoBack: () => void;
  isLoading: boolean;
  results: ArtResult[];
  responseText: string;
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
  results,
  responseText,
  currentTime,
  searchInputRef,
  isFadingOut,
}) => (
  <div
    className={`main-container flex flex-col p-4 sm:p-6 md:p-8 relative transition-opacity duration-500 ${
      isFadingOut ? "opacity-0" : "opacity-100"
    }`}
  >
    <header className="w-full flex justify-start items-center z-10">
      <button
        onClick={onGoBack}
        className="w-10 h-10 bg-stone-800/50 rounded-full flex items-center justify-center cursor-pointer"
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
    <main className="flex-grow flex flex-col items-center justify-start w-full max-w-6xl mx-auto pt-16">
      <div className="w-full">
        <p className="text-2xl mt-4 text-center text-stone-100 ">
          {responseText}
        </p>
        <input
          ref={searchInputRef}
          type="text"
          value={inputValue}
          onChange={onInputChange}
          onKeyDown={onSearch}
          placeholder="A futuristic city skyline at dusk, impressionist style..."
          className="w-full bg-transparent border-b border-gray-600 text-2xl md:text-4xl text-center text-stone-100 focus:outline-none focus:border-stone-100 transition-colors py-4"
        />
        <p className="text-center text-gray-500 text-sm mt-4">
          Press Enter to curate
        </p>
      </div>

      {isLoading && <LoaderIcon />}

      {results.length > 0 && (
        <div className="w-full mt-12">
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6">
            {results.map((item, index) => (
              <ArtCard key={index} item={item} index={index} />
            ))}
          </div>
        </div>
      )}
    </main>
    <Footer currentTime={currentTime} />
  </div>
);

export default InputPage;
