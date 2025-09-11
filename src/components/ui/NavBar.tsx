import React from "react";

interface NavBarProps {
  isOpen: boolean;
  onClose: () => void;
  menuRef: React.RefObject<HTMLDivElement | null>;
}

const NavBar: React.FC<NavBarProps> = ({ isOpen, onClose, menuRef }) => (
  <div
    ref={menuRef}
    className={`mobile-menu fixed top-0 right-0 w-80 h-full bg-stone-900/95 backdrop-blur-sm z-50 p-8 ${
      isOpen ? "open" : ""
    }`}
  >
    <div className="flex justify-end mb-12">
      <button
        onClick={onClose}
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
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
    </div>
    <nav className="space-y-8">
      {["Features", "How It Works"].map((item) => (
        <a
          key={item}
          href={`#${item.toLowerCase().replace(" ", "-")}`}
          onClick={onClose}
          className="block text-2xl font-bold hover:text-gray-300 transition-colors"
        >
          {item}
        </a>
      ))}
    </nav>
  </div>
);

export default NavBar;
