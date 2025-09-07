// src/components/sections/Gallery.tsx
import React, { useEffect, useRef } from 'react';

// Mock data for a more realistic gallery
const galleryItems = [
    { id: 1, seed: 'sky', artist: 'Gaspar', title: 'Celestial Dream' },
    { id: 2, seed: 'portrait', artist: 'Michael', title: 'The Observer' },
    { id: 3, seed: 'abstract', artist: 'Gaspar', title: 'Chroma Burst' },
    { id: 4, seed: 'nature', artist: 'Michael', title: 'Verdant Whisper' },
    { id: 5, seed: 'city', artist: 'Gaspar', title: 'Urban Pulse' },
    { id: 6, seed: 'water', artist: 'Michael', title: 'Azure Depths' },
    { id: 7, seed: 'texture', artist: 'Gaspar', title: 'Woven Earth' },
    { id: 8, seed: 'minimal', artist: 'Michael', title: 'Solitude' },
    { id: 9, seed: 'space', artist: 'Gaspar', title: 'Nebula' },
    { id: 10, seed: 'animal', artist: 'Michael', title: 'Silent' },
    { id: 11, seed: 'sculpture', artist: 'Gaspar', title: 'Fluid' },
];

const Gallery: React.FC = () => {
    const gridRef = useRef<HTMLDivElement>(null);
    const headingRef = useRef<HTMLHeadingElement>(null);

    // Effect for animating grid items
    useEffect(() => {
        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('is-visible');
                        observer.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.1 }
        );

        // Observe the heading
        if (headingRef.current) {
            observer.observe(headingRef.current);
        }
        
        // Observe each gallery item
        const elements = gridRef.current?.querySelectorAll('.gallery-item');
        elements?.forEach((el) => observer.observe(el));

        return () => observer.disconnect();
    }, []);

    return (
        <section id="gallery" className="min-h-screen bg-stone-900 text-stone-200 p-4 sm:p-6 md:p-8 flex items-center">
            <div className="max-w-7xl mx-auto w-full">
                <h2 
                    ref={headingRef}
                    className="gallery-item font-brand text-6xl md:text-8xl lg:text-9xl mb-16 text-center text-stone-500"
                >
                    GALLERY
                </h2>

                {/* Masonry Layout Container */}
                <div ref={gridRef} className="columns-2 md:columns-3 lg:columns-4 gap-4 md:gap-6">
                    {galleryItems.map((item, i) => (
                        <div 
                            key={item.id} 
                            className="gallery-item mb-4 md:mb-6 break-inside-avoid group relative rounded-lg overflow-hidden shadow-lg"
                            style={{ transitionDelay: `${i * 70}ms` }}
                        >
                            {/* eslint-disable-next-line @next/next/no-img-element */}
                            <img
                                src={`https://picsum.photos/seed/${item.seed}/600/800`}
                                alt={item.title}
                                className="w-full h-auto object-cover transition-transform duration-500 ease-in-out group-hover:scale-110"
                            />
                            {/* Interactive Overlay */}
                            <div className="absolute inset-0 bg-black/70 flex flex-col justify-end p-6 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                <h3 className="font-bold text-lg text-white transform-gpu translate-y-4 group-hover:translate-y-0 transition-transform duration-300">{item.title}</h3>
                                <p className="text-stone-300 text-sm transform-gpu translate-y-4 group-hover:translate-y-0 transition-transform duration-300 delay-75">{item.artist}</p>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="text-center mt-16">
                    <p className="text-stone-400 text-lg mb-8 max-w-md mx-auto">Discover thousands of curated artworks from talented artists worldwide.</p>
                    <button className="bg-stone-800/80 hover:bg-stone-700/80 border border-stone-700/80 transition-all duration-300 px-10 py-4 rounded-full font-bold text-stone-200 hover:text-white hover:shadow-lg hover:shadow-stone-500/10 transform hover:-translate-y-1">
                        EXPLORE MORE
                    </button>
                </div>
            </div>
        </section>
    );
};

export default Gallery;