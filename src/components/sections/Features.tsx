// src/components/sections/Features.tsx
import React, { useEffect, useRef } from 'react';

// Define the icons as separate components for clarity
const AICurationIcon = () => (
    <svg xmlns="http://www.w.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 12l2 2 4-4"/><path d="M21 12c.552 0 1-.448 1-1V5c0-.552-.448-1-1-1H3c-.552 0-1 .448-1 1v6c0 .552.448 1 1 1"/><path d="M3 12v6c0 .552.448 1 1 1h16c.552 0 1-.448 1-1v-6"/></svg>
);

const HumanMadeIcon = () => (
    <svg xmlns="http://www.w.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v6m11-7h-6m-6 0H1"/></svg>
);

const PersonalizedIcon = () => (
    <svg xmlns="http://www.w.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
);

const featuresData = [
    {
        icon: <AICurationIcon />,
        title: "AI-POWERED CURATION",
        description: "Advanced algorithms analyze your preferences to discover art that matches your unique taste profile."
    },
    {
        icon: <HumanMadeIcon />,
        title: "HUMAN-MADE FOCUS",
        description: "Exclusively featuring authentic human-created artwork from talented artists around the world."
    },
    {
        icon: <PersonalizedIcon />,
        title: "PERSONALIZED DISCOVERY",
        description: "Every recommendation is tailored to your evolving artistic preferences and discovery journey."
    }
];

const Features: React.FC = () => {
    const sectionRef = useRef<HTMLDivElement>(null);

    // This useEffect handles the scroll-triggered animations
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('animate-fade-in-up');
                        observer.unobserve(entry.target); // Animate only once
                    }
                });
            },
            { threshold: 0.1 } // Trigger when 10% of the element is visible
        );

        const elements = sectionRef.current?.querySelectorAll('.feature-card, .section-heading');
        elements?.forEach((el) => observer.observe(el));

        return () => observer.disconnect();
    }, []);


    return (
        <section 
            id="features" 
            className="relative min-h-screen bg-stone-900 text-stone-200 p-4 sm:p-6 md:p-8 flex items-center justify-center overflow-hidden"
        >
            {/* Subtle decorative background gradient */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60rem] h-[60rem] bg-radial-gradient from-stone-800/40 to-transparent rounded-full blur-3xl pointer-events-none"></div>

            <div ref={sectionRef} className="max-w-6xl mx-auto w-full z-10">
                <h2 className="section-heading font-brand text-6xl md:text-8xl lg:text-9xl mb-24 text-center text-stone-500 opacity-0">
                    FEATURES
                </h2>

                <div className="grid md:grid-cols-3 gap-12 md:gap-16">
                    {featuresData.map((feature, index) => (
                        <div 
                            key={index} 
                            className="feature-card text-center group opacity-0"
                            style={{ animationDelay: `${index * 200}ms` }}
                        >
                            <div className="w-20 h-20 bg-stone-800/50 border border-stone-700/80 rounded-full flex items-center justify-center mx-auto mb-8 transition-all duration-300 group-hover:scale-110 group-hover:bg-stone-700/70 group-hover:shadow-lg group-hover:shadow-stone-500/10">
                                {feature.icon}
                            </div>
                            <h3 className="font-bold text-xl mb-4 text-stone-100">{feature.title}</h3>
                            <p className="text-stone-400 leading-relaxed">{feature.description}</p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default Features;