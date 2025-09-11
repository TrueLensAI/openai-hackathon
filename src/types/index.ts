export interface ArtResult {
  title: string;
  artist: string;
  match: number;
  id: string;
}

export interface Message {
  id: number;
  type: "user" | "assistant";
  content: string;
  timestamp: Date;
  images: Image[] | null;
  suggestions?: string[] | null;
  processingTime?: number;
}

export interface Image {
  image_url: string;
  title: string;
  similarity_score: number;
  metadata?: {
    confidence: string;
  };
  marketplace?: string;
  price?: string;
  page_url: string;
  purchase_steps?: string[];
}
