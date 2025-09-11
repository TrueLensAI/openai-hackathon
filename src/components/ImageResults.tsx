import React from "react";
import type { Image } from "../types/index";

// --- SVG Icon Components (replaces lucide-react) ---

const ExternalLink = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
    <polyline points="15 3 21 3 21 9" />
    <line x1="10" y1="14" x2="21" y2="3" />
  </svg>
);

const ShoppingCart = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <circle cx="9" cy="21" r="1" />
    <circle cx="20" cy="21" r="1" />
    <path d="M1 1h4l2.68 13.39a2 2 0 0 0 2 1.61h9.72a2 2 0 0 0 2-1.61L23 6H6" />
  </svg>
);

const Star = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2" />
  </svg>
);

const DollarSign = ({ className }: { className?: string }) => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <line x1="12" y1="1" x2="12" y2="23" />
      <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
    </svg>
);


const ImageResults = ({ images }: { images: Image[] | null }) => {
  console.log("Images:", images);

  if (!images || images.length === 0) {
    return null;
  }

  const getConfidenceColor = (confidence: string) => {
    switch (confidence) {
      case "high":
        return "bg-green-900/50 text-green-300";
      case "medium":
        return "bg-yellow-900/50 text-yellow-300";
      case "low":
        return "bg-stone-700 text-stone-300";
      default:
        return "bg-blue-900/50 text-blue-300";
    }
  };

  const getConfidenceEmoji = (confidence: string) => {
    switch (confidence) {
      case "high":
        return "🔥";
      case "medium":
        return "✨";
      case "low":
        return "💫";
      default:
        return "🎨";
    }
  };

  return (
    <div className="mt-6 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
      {images.map((image: Image, index: number) => (
        <div
          key={index}
          className="bg-stone-800/50 rounded-xl shadow-lg overflow-hidden border border-stone-700 hover:shadow-xl transition-shadow"
        >
          {/* Image */}
          <div className="relative">
            <img
              src={image.image_url}
              alt={image.title}
              className="w-full h-48 object-cover"
              onError={(e: React.SyntheticEvent<HTMLImageElement>) => {
                (e.target as HTMLImageElement).src =
                  "https://placehold.co/600x400/262626/FFFFFF?text=Image+Not+Available";
              }}
            />
            {/* Similarity Score Badge */}
            <div className="absolute top-3 right-3 bg-stone-900/80 text-stone-100 px-3 py-1 rounded-full text-sm font-semibold flex items-center">
              <Star className="w-4 h-4 mr-1 fill-current" />
              {Math.round(image.similarity_score * 100)}%
            </div>
            {/* Confidence Badge */}
            <div
              className={`absolute top-3 left-3 px-2 py-1 rounded-full text-xs font-medium ${getConfidenceColor(
                image.metadata?.confidence || "match"
              )}`}
            >
              {getConfidenceEmoji(image.metadata?.confidence || "match")}{" "}
              {image.metadata?.confidence || "match"}
            </div>
          </div>

          {/* Content */}
          <div className="p-5">
            <h3 className="font-bold text-lg text-stone-100 mb-2 line-clamp-2">
              {image.title}
            </h3>

            {/* Marketplace and Price */}
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm text-stone-400">
                📍 {image.marketplace || "Marketplace"}
              </div>
              {image.price && (
                <div className="flex items-center text-green-400 font-semibold">
                  <DollarSign className="w-4 h-4" />
                  {image.price.replace("$", "")}
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-2 mb-4">
              <a
                href={image.page_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 bg-blue-700 hover:bg-blue-800 text-stone-100 px-4 py-2 rounded-lg text-sm font-medium flex items-center justify-center transition-colors"
              >
                <ExternalLink className="w-4 h-4 mr-2" />
                View
              </a>
              <a
                href={image.page_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex-1 bg-green-700 hover:bg-green-800 text-stone-100 px-4 py-2 rounded-lg text-sm font-medium flex items-center justify-center transition-colors"
              >
                <ShoppingCart className="w-4 h-4 mr-2" />
                Buy Now
              </a>
            </div>

            {/* Purchase Steps */}
            {image.purchase_steps && image.purchase_steps.length > 0 && (
              <details className="text-sm">
                <summary className="cursor-pointer text-stone-400 hover:text-stone-200 font-medium">
                  📋 How to purchase
                </summary>
                <ol className="mt-2 text-stone-400 list-decimal list-inside space-y-1 text-xs">
                  {image.purchase_steps.map(
                    (step: string, stepIndex: number) => (
                      <li key={stepIndex}>{step}</li>
                    )
                  )}
                </ol>
              </details>
            )}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ImageResults;
