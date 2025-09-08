import React from 'react';

interface HeaderProps {
    onMenuOpen: () => void;
    menuButtonRef: React.RefObject<HTMLDivElement | null>;
}

const Header: React.FC<HeaderProps> = ({ onMenuOpen, menuButtonRef }) => (
    <header className="w-full flex justify-between items-center z-10">
        <div className="w-10 h-10 bg-white rounded-full flex items-center justify-center">
            <svg xmlns="http://www.w3.org/2000/svg" height="20" width="20" viewBox="0 0 512 512" className="w-5 h-5">
                <path fill="#111" d="M416 208c0 45.9-14.9 88.3-40 122.7L502.6 457.4c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L330.7 376C296.3 401.1 253.9 416 208 416 93.1 416 0 322.9 0 208S93.1 0 208 0 416 93.1 416 208zM208 352a144 144 0 1 0 0-288 144 144 0 1 0 0 288z"/>
            </svg>
        </div>
        <div ref={menuButtonRef} onClick={onMenuOpen} className="w-10 h-10 bg-stone-800/50 rounded-full flex items-center justify-center cursor-pointer">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f5f5f4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
        </div>
    </header>
);

export default Header;