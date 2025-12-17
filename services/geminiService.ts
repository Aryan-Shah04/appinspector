import { GoogleGenAI } from "@google/genai";
import { AppSearchResult, AppAnalysis, ChatMessage } from '../types';

// Declare process for Vite environment variable usage in TS
declare const process: { env: { API_KEY: string } };

// API Key must be obtained from environment variables
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

// Model Constants
// Updated to the specific dialog model name as requested
const MODEL_NAME = "gemini-2.5-flash-native-audio-dialog";

// Token Management
const MAX_CONTEXT_CHARS = 60000;

const manageContextWindow = (history: ChatMessage[], systemContext: string): ChatMessage[] => {
  let currentChars = systemContext.length;
  const safeHistory: ChatMessage[] = [];
  for (let i = history.length - 1; i >= 0; i--) {
    const msg = history[i];
    const msgLen = msg.content.length;
    if (currentChars + msgLen < MAX_CONTEXT_CHARS) {
      safeHistory.unshift(msg);
      currentChars += msgLen;
    } else {
      break;
    }
  }
  return safeHistory;
};

const extractJson = <T>(text: string): T | null => {
  try {
    let cleanText = text.trim();
    // Remove markdown code blocks if present
    const match = cleanText.match(/```json\s*([\s\S]*?)\s*```/) || cleanText.match(/```\s*([\s\S]*?)\s*```/);
    if (match && match[1]) cleanText = match[1];
    
    // Clean up any potential leading/trailing non-JSON characters
    const firstOpen = cleanText.indexOf('{');
    const firstArray = cleanText.indexOf('[');
    
    if (firstOpen !== -1 && (firstArray === -1 || firstOpen < firstArray)) {
       const lastClose = cleanText.lastIndexOf('}');
       if (lastClose !== -1) cleanText = cleanText.substring(firstOpen, lastClose + 1);
    } else if (firstArray !== -1) {
       const lastClose = cleanText.lastIndexOf(']');
       if (lastClose !== -1) cleanText = cleanText.substring(firstArray, lastClose + 1);
    }

    return JSON.parse(cleanText);
  } catch (error) {
    console.warn("JSON extraction failed:", error);
    return null;
  }
};

const cleanData = (analysis: AppAnalysis, fallbackRating?: string): AppAnalysis => {
  const cleanRating = (r: string) => {
    // 1. Try to extract a valid float rating
    const match = r ? r.match(/([0-5]\.\d)/) : null;
    if (match) return match[0];
    
    // 2. Fallback to the search result rating
    if (fallbackRating && 
        fallbackRating !== "N/A" && 
        /^[0-5](\.\d)?$/.test(fallbackRating)) {
      return fallbackRating;
    }

    // 3. Fallback for integer ratings
    if (r && /^[1-5]$/.test(r)) return r + ".0";

    return "N/A";
  };

  const cleanDownloads = (d: string) => {
    if (!d || d === "N/A") return "N/A";
    let val = d.replace(/downloads|over|approx|more than|installations|installs/gi, '').trim();

    if (val.match(/million/i)) val = val.replace(/million/i, 'M');
    if (val.match(/billion/i)) val = val.replace(/billion/i, 'B');
    if (val.match(/thousand/i)) val = val.replace(/thousand/i, 'k');

    const match = val.match(/(\d{1,3}(,\d{3})+(\+)?)|(\d+(\.\d+)?\s*[MBK]\+?)|(\d+\+)|(\d{3,}\+?)/i);
    
    if (match) {
      return match[0].toUpperCase().replace(/\s/g, '');
    }
    return "N/A";
  };

  const cleanDate = (d: string) => {
    if (!d) return undefined; 
    const match = d.match(/([A-Z][a-z]{2,}\s\d{1,2},\s\d{4})|(\d{4}-\d{2}-\d{2})|(\d{1,2}\s[A-Z][a-z]{2,}\s\d{4})/i);
    if (match) return match[0];
    return undefined;
  };

  return {
    ...analysis,
    reviewSummary: analysis.reviewSummary || "No review summary could be generated.",
    authenticity: analysis.authenticity || "Authenticity check could not be completed.",
    background: analysis.background || "Developer background information unavailable.",
    rating: cleanRating(analysis.rating),
    downloads: cleanDownloads(analysis.downloads),
    lastUpdated: cleanDate(analysis.lastUpdated || "")
  };
};

export const searchApps = async (query: string): Promise<AppSearchResult[]> => {
  try {
    const searchPrompt = `site:play.google.com/store/apps/details ${query}`;
    
    const response = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: `
        Perform a Google Search for: "${searchPrompt}".
        
        Task: Find REAL Android apps listed on the Google Play Store (play.google.com).
        
        Output strictly a JSON Array.
        Format:
        [
          {
            "name": "App Name",
            "developer": "Developer Name",
            "description": "Short description",
            "rating": "4.5" 
          }
        ]
      `,
      config: {
        tools: [{ googleSearch: {} }] 
      }
    });

    const data = extractJson<AppSearchResult[]>(response.text || "[]");
    return (data || []).slice(0, 4).map(app => ({
      ...app,
      rating: app.rating ? (app.rating.match(/([0-5](\.\d)?)/)?.[0] || "N/A") : "N/A"
    }));
  } catch (error) {
    console.error("Error searching apps:", error);
    throw error;
  }
};

export const analyzeApp = async (app: AppSearchResult): Promise<AppAnalysis> => {
  try {
    const response = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: `
        Analyze the Android app "${app.name}" by "${app.developer}".
        Use Google Search to find its Play Store page.

        Extract:
        1. Rating (e.g. 4.5)
        2. Downloads (e.g. 100M+)
        3. Last Updated Date
        4. Review Summary (User sentiment)
        5. Authenticity (Is it official?)
        6. Developer Background

        Return strictly valid JSON:
        {
          "reviewSummary": "string",
          "authenticity": "string",
          "background": "string",
          "rating": "string",
          "downloads": "string",
          "lastUpdated": "string"
        }
      `,
      config: {
        tools: [{ googleSearch: {} }]
      }
    });

    const rawAnalysis = extractJson<AppAnalysis>(response.text || "{}");
    
    if (!rawAnalysis) throw new Error("Could not parse analysis");

    const groundingUrls: string[] = [];
    const chunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
    if (chunks) {
      chunks.forEach((chunk: any) => {
        if (chunk.web?.uri) groundingUrls.push(chunk.web.uri);
      });
    }

    const cleanAnalysis = cleanData(rawAnalysis, app.rating);

    return {
      ...cleanAnalysis,
      groundingUrls: Array.from(new Set(groundingUrls))
    };
    
  } catch (error) {
    console.error("Error analyzing app:", error);
    throw error;
  }
};

export const chatWithApp = async (
  history: ChatMessage[], 
  newMessage: string, 
  appContext: AppSearchResult, 
  analysisContext: AppAnalysis
): Promise<string> => {
  
  const systemInstruction = `
    You are an app safety assistant.
    App: "${appContext.name}" by "${appContext.developer}".
    Stats: Rating ${analysisContext.rating}, Downloads ${analysisContext.downloads}.
    Reviews: ${analysisContext.reviewSummary}
    Safety: ${analysisContext.authenticity}
    
    Keep answers concise and helpful.
  `;

  const cleanHistory = manageContextWindow(history, systemInstruction + newMessage);

  const contents = [
    ...cleanHistory.map(msg => ({
      role: msg.role,
      parts: [{ text: msg.content }]
    })),
    {
      role: 'user',
      parts: [{ text: newMessage }]
    }
  ];

  try {
    const response = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: contents,
      config: {
        systemInstruction: systemInstruction,
        // Ensure we are only asking for text back
        responseMimeType: "text/plain"
      }
    });

    return response.text || "No response generated.";
  } catch (error) {
    console.error("Chat error:", error);
    throw error;
  }
};