import React from "react";
import type { ArtResult } from "../../types";

interface ArtCardProps {
  item: ArtResult;
  index: number;
}

const ArtCard: React.FC<ArtCardProps> = ({ item, index }) => {
  const aspects = ["aspect-square", "aspect-[3/4]", "aspect-[4/3]"];
  const randomAspect = aspects[index % aspects.length];

  return (
    <div className="art-card group relative overflow-hidden rounded-lg cursor-pointer transform hover:scale-105 transition-transform duration-300">
      <img
        src={item.imageUrl}
        alt={item.title}
        className={`w-full h-full object-cover ${randomAspect}`}
        onError={(e) =>
          (e.currentTarget.src =
            "https://placehold.co/800x600/1c1917/f5f5f4?text=Art+Not+Found")
        }
      />
      <div className="overlay absolute inset-0 bg-black/70 flex flex-col justify-end p-4">
        <h3 className="font-bold text-lg">{item.title}</h3>
        <p className="text-sm text-gray-300">by {item.artist}</p>
        <div className="mt-2 text-xs font-bold bg-stone-800/80 text-white py-1 px-2 rounded-full self-start">
          {item.match}% Match
        </div>
      </div>
    </div>
  );
};

export default ArtCard;
