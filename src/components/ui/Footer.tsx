import React from 'react';

interface FooterProps {
    currentTime: string;
    showSocials?: boolean;
}

const Footer: React.FC<FooterProps> = ({ currentTime, showSocials = false }) => (
    <footer className={`w-full flex items-center z-10 text-sm ${showSocials ? 'justify-between' : 'justify-end'}`}>
        {showSocials && (
            <div>
                <span className="font-bold">SOCIALS:</span>
                <span>@TRUE_LENS</span>
            </div>
        )}
        <div className="font-mono text-right">{currentTime}</div>
    </footer>
);

export default Footer;