// src/components/pages/LandingPage.tsx
import React from 'react';
import Header from '../ui/Header';
import Footer from '../ui/Footer';
import Features from '../sections/Features';
import HowItWorks from '../sections/HowItWorks';
import Gallery from '../sections/Gallery';

interface LandingPageProps {
    currentTime: string;
    onMenuOpen: () => void;
    menuButtonRef: React.RefObject<HTMLDivElement | null>;
    isFadingOut: boolean;
    goToInputPage: (startingChar?: string) => void; // new
}

const LandingPage: React.FC<LandingPageProps> = ({
    currentTime,
    onMenuOpen,
    menuButtonRef,
    isFadingOut,
    goToInputPage
}) => (
    <>
        <div className={`main-container flex flex-col p-4 sm:p-6 md:p-8 relative transition-opacity duration-500 ${isFadingOut ? 'opacity-0' : 'opacity-100'}`}>
            <Header onMenuOpen={onMenuOpen} menuButtonRef={menuButtonRef} />
            <main className="flex-grow flex items-center justify-center relative">
                <h1 className="font-brand text-[15vw] sm:text-[15vw] md:text-[180px] lg:text-[250px] absolute top-[45%] left-1/2 -translate-x-1/2 -translate-y-1/2 text-center leading-none select-none" style={{ color: '#222' }}>
                    TRUE LENS
                </h1>
            </main>
            <div className="hidden md:block absolute top-1/2 -translate-y-1/2 left-8 lg:left-16 z-10 max-w-[200px]">
                <p className="font-bold text-lg leading-tight">THE AI CURATOR THAT HELPS YOU DISCOVER A THING OR TWO.</p>
                <p className="mt-4 text-lg leading-tight">STARTING WITH YOUR TASTE.</p>
            </div>
            <div className="hidden md:block absolute top-1/2 -translate-y-1/2 right-8 lg:right-16 z-10 max-w-[200px] text-right">
                <p className="font-bold text-lg leading-tight">FIND THE BEST HUMAN-MADE ART FOR YOU.</p>
            </div>
            {/* Clickable "START TYPING" */}
            <div
                className="prompt-container absolute bottom-24 left-1/2 -translate-x-1/2 text-gray-500 text-sm tracking-widest z-10 cursor-pointer"
                onClick={() => goToInputPage()} // trigger same as typing
            >
                START TYPING
            </div>
            <Footer currentTime={currentTime} showSocials={true} />
        </div>
        <Features />
        <HowItWorks />
        <Gallery />
    </>
);

export default LandingPage;
