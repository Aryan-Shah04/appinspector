export interface AppSearchResult {
  name: string;
  developer: string;
  description: string;
  rating?: string;
}

export interface AppAnalysis {
  reviewSummary: string;
  authenticity: string;
  background: string;
  rating: string;
  downloads: string;
  lastUpdated?: string;
  iconUrl?: string;
  groundingUrls?: string[];
}

export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
}

export enum AppState {
  SEARCH = 'SEARCH',
  SELECTING = 'SELECTING',
  ANALYZING = 'ANALYZING',
  DASHBOARD = 'DASHBOARD'
}