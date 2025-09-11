// src/App.tsx
import React, { useState, useEffect, useRef } from "react";
import type { ArtResult, Message } from "./types";
import NavBar from "./components/ui/NavBar";
import LandingPage from "./components/pages/LandingPage";
import InputPage from "./components/pages/InputPage";
import { useImageSearch } from "./hooks/useImageSearch";

const App: React.FC = () => {
  const [page, setPage] = useState<"landing" | "input">("landing");
  const [isFadingOut, setIsFadingOut] = useState(false);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<ArtResult[]>([]);
  const [currentTime, setCurrentTime] = useState("");
  const { messages, sendMessage, isLoading: searchLoading } = useImageSearch();

  const searchInputRef = useRef<HTMLInputElement>(null);
  const menuButtonRef = useRef<HTMLDivElement>(null);
  const navBarRef = useRef<HTMLDivElement>(null);

  // --- Clock ---
  useEffect(() => {
    const updateClock = () => {
      const now = new Date();
      setCurrentTime(
        now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
      );
    };
    updateClock();
    const timerId = setInterval(updateClock, 60000);
    return () => clearInterval(timerId);
  }, []);

  // --- Landing page typing ---
  useEffect(() => {
    if (page !== "landing") return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        !isMenuOpen &&
        event.key.length === 1 &&
        !event.ctrlKey &&
        !event.altKey &&
        !event.metaKey
      ) {
        event.preventDefault();
        goToInputPage(event.key); // first character
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [page, isMenuOpen]);

  // --- Close menu on outside click ---
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        isMenuOpen &&
        navBarRef.current &&
        !navBarRef.current.contains(event.target as Node) &&
        menuButtonRef.current &&
        !menuButtonRef.current.contains(event.target as Node)
      ) {
        setIsMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isMenuOpen]);

  // --- Navigation ---
  const goToInputPage = (startingChar: string = "") => {
    setIsFadingOut(true);
    setTimeout(() => {
      setPage("input");
      setInputValue(startingChar);
      setIsFadingOut(false);
      setTimeout(() => searchInputRef.current?.focus(), 0);
    }, 500);
  };

  const goBackToLanding = () => {
    setIsFadingOut(true);
    setTimeout(() => {
      setPage("landing");
      setInputValue("");
      setResults([]);
      setIsLoading(false);
      setIsFadingOut(false);
    }, 500);
  };

  const goToEthos = () => {
    // Placeholder for ethos navigation
    console.log("Navigate to Ethos");
  };

  // --- Search ---
  const handleSearch = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter" && inputValue.trim() !== "") {
      event.preventDefault();
      fetchResults();
    }
  };

  const fetchResults = async () => {
    setResults([]);
    setIsLoading(true);
    try {
      await sendMessage(inputValue);
      // The images will be in the latest assistant message
      // We'll handle this in InputPage
    } catch (error) {
      console.error("Search failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{ fontFamily: "'Inter', sans-serif" }}
      className="antialiased bg-[#111] text-[#f5f5f4] overflow-x-hidden relative min-h-screen"
    >
      <style>{`
                .font-brand { font-family: 'Archivo Black', sans-serif; }
                .main-container { min-height: 100vh; width: 100vw; }
                .prompt-container::after { content: '_'; opacity: 1; animation: blink 1s infinite; }
                @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
                .mobile-menu { transform: translateX(100%); transition: transform 0.3s ease-in-out; }
                .mobile-menu.open { transform: translateX(0); }
                html { scroll-behavior: smooth; }
                .art-card .overlay { opacity: 0; transition: opacity 0.3s ease-in-out; }
                .art-card:hover .overlay { opacity: 1; }
            `}</style>

      <NavBar
        isOpen={isMenuOpen}
        onClose={() => setIsMenuOpen(false)}
        menuRef={navBarRef}
      />

      {/* Landing Page */}
      <div
        className={`absolute inset-0 transition-opacity duration-500 ${
          page === "landing" && !isFadingOut
            ? "opacity-100 z-10"
            : "opacity-0 z-0 pointer-events-none"
        }`}
      >
        <LandingPage
          currentTime={currentTime}
          onMenuOpen={() => setIsMenuOpen(true)}
          menuButtonRef={menuButtonRef}
          isFadingOut={isFadingOut}
          goToInputPage={goToInputPage}
        />
      </div>

      {/* Input Page */}
      <div
        className={`absolute inset-0 transition-opacity duration-500 ${
          page === "input" && !isFadingOut
            ? "opacity-100 z-10"
            : "opacity-0 z-0 pointer-events-none"
        }`}
      >
        <InputPage
          inputValue={inputValue}
          onInputChange={(e) => setInputValue(e.target.value)}
          onSearch={handleSearch}
          onGoBack={goBackToLanding}
          isLoading={isLoading}
          messages={messages}
          currentTime={currentTime}
          searchInputRef={searchInputRef}
          isFadingOut={isFadingOut}
        />
      </div>
    </div>
  );
};

export default App;
