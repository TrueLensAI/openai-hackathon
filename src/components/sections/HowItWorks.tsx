import React from 'react';

const HowItWorks: React.FC = () => (
    <section id="how-it-works" className="min-h-screen bg-black p-4 sm:p-6 md:p-8 flex items-center">
                         <div className="max-w-6xl mx-auto w-full">
                             <h2 className="font-brand text-6xl md:text-8xl lg:text-9xl mb-16 text-center" style={{color: '#222'}}>HOW IT WORKS</h2>
                             <div className="space-y-20">
                                <div className="flex flex-col md:flex-row items-center gap-12">
                                    <div className="md:w-1/2">
                                        <div className="text-6xl font-brand mb-4" style={{color: '#333'}}>01</div>
                                        <h3 className="font-bold text-2xl mb-4">DESCRIBE YOUR VISION</h3>
                                        <p className="font-sans text-gray-400 text-lg leading-relaxed">Start by typing what you're looking for. Whether it's a mood, style, or specific subject matter, our AI understands your creative language.</p>
                                    </div>
                                    <div className="md:w-1/2 bg-stone-900/50 rounded-lg p-8 text-center">
                                        <div className="text-gray-500 text-sm mb-2">EXAMPLE INPUT</div>
                                        <div className="text-xl italic">"Moody urban landscapes with warm lighting"</div>
                                    </div>
                                </div>
                                <div className="flex flex-col md:flex-row-reverse items-center gap-12">
                                    <div className="md:w-1/2">
                                        <div className="text-6xl font-brand mb-4" style={{color: '#333'}}>02</div>
                                        <h3 className="font-bold text-2xl mb-4">AI ANALYSIS</h3>
                                        <p className="text-gray-400 text-lg leading-relaxed">Our sophisticated algorithms analyze thousands of artworks, matching style, composition, color palette, and emotional resonance to your request.</p>
                                    </div>
                                    <div className="md:w-1/2 bg-stone-900/50 rounded-lg p-8 flex justify-center">
                                        <div className="w-16 h-16 border-2 border-gray-600 rounded-full animate-spin border-t-white"></div>
                                    </div>
                                </div>
                                <div className="flex flex-col md:flex-row items-center gap-12">
                                    <div className="md:w-1/2">
                                        <div className="text-6xl font-brand mb-4" style={{color: '#333'}}>03</div>
                                        <h3 className="font-bold text-2xl mb-4">CURATED RESULTS</h3>
                                        <p className="text-gray-400 text-lg leading-relaxed">Receive a carefully curated selection of human-made artworks that perfectly match your aesthetic preferences and creative vision.</p>
                                    </div>
                                    <div className="md:w-1/2 grid grid-cols-2 gap-4">
                                        <div className="bg-stone-800 rounded-lg h-24"></div>
                                        <div className="bg-stone-700 rounded-lg h-24"></div>
                                        <div className="bg-stone-600 rounded-lg h-24"></div>
                                        <div className="bg-stone-800 rounded-lg h-24"></div>
                                    </div>
                                </div>
                            </div>
                         </div>
                    </section>
);
export default HowItWorks;