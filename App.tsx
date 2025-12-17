import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import SearchResults from './components/SearchResults';
import Dashboard from './components/Dashboard';
import { AppState, AppSearchResult, AppAnalysis } from './types';
import { searchApps, analyzeApp } from './services/geminiService';
import { Loader2, ShieldCheck, Download, X, HelpCircle, Github } from 'lucide-react';
import JSZip from 'jszip';

const App: React.FC = () => {
  const [appState, setAppState] = useState<AppState>(AppState.SEARCH);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [searchResults, setSearchResults] = useState<AppSearchResult[]>([]);
  const [selectedApp, setSelectedApp] = useState<AppSearchResult | null>(null);
  const [analysis, setAnalysis] = useState<AppAnalysis | null>(null);
  
  const [showDownloadModal, setShowDownloadModal] = useState(false);

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    setError(null);
    setAppState(AppState.SEARCH);
    setSearchResults([]);

    try {
      const results = await searchApps(query);
      if (results.length === 0) {
        setError("No apps found. The AI might be busy, please try again.");
      } else {
        setSearchResults(results);
        setAppState(AppState.SELECTING);
      }
    } catch (err) {
      setError("Failed to search. The AI encountered an error. Please try again.");
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectApp = async (app: AppSearchResult) => {
    setSelectedApp(app);
    setAppState(AppState.ANALYZING);
    setIsLoading(true);
    setError(null);

    try {
      const analysisData = await analyzeApp(app);
      setAnalysis(analysisData);
      setAppState(AppState.DASHBOARD);
    } catch (err) {
      setError("Analysis failed. Please try selecting the app again.");
      setAppState(AppState.SELECTING);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleBackToSearch = () => {
    setAppState(AppState.SEARCH);
    setSearchResults([]);
    setSelectedApp(null);
    setAnalysis(null);
    setError(null);
  };

  const downloadSourceCode = async () => {
    try {
      const zip = new JSZip();
      
      // 1. Fetch Source Files
      const filesToDownload = [
        'index.tsx',
        'App.tsx',
        'types.ts',
        'services/geminiService.ts',
        'components/SearchBar.tsx',
        'components/SearchResults.tsx',
        'components/Dashboard.tsx',
        'components/MarkdownText.tsx',
        'metadata.json'
      ];

      for (const file of filesToDownload) {
        try {
          const response = await fetch(file);
          if (response.ok) {
             const text = await response.text();
             zip.file(file, text);
          }
        } catch (e) {
          console.warn(`Error fetching ${file}:`, e);
        }
      }

      // 2. Add Vite Configuration
      zip.file('vite.config.ts', `
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: './', 
  define: {
    'process.env.API_KEY': JSON.stringify("AIzaSyCZSqy5CN2DSbm0BWii3k7z4qtrhitzB8Q")
  }
})
      `.trim());

      // 3. Add TS Config
      zip.file('tsconfig.json', `
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true
  },
  "include": ["**/*"],
  "exclude": ["node_modules"]
}
      `.trim());

      // 4. Custom Deploy Script to fix Windows ENAMETOOLONG issues
      zip.file('deploy.js', `
import ghpages from 'gh-pages';

console.log('🚀 Starting deployment to GitHub Pages...');

const options = {
  branch: 'gh-pages',
  repo: undefined, // uses 'origin' by default
  // Setting history to false helps avoid "ENAMETOOLONG" errors on Windows
  // by preventing the library from cloning the entire git history into a deep cache folder.
  history: false, 
  dotfiles: true
};

ghpages.publish('dist', options, (err) => {
  if (err) {
    console.error('❌ Deployment Failed:', err);
    console.log('\\n--- 🛠️ MANUAL FALLBACK INSTRUCTIONS ---');
    console.log('The automatic deploy script failed (this is common on Windows due to path limits).');
    console.log('Don\\'t worry! You can deploy manually by running these 5 commands in your terminal:\\n');
    console.log('  1. git checkout -b gh-pages');
    console.log('  2. npm run build');
    console.log('  3. git add dist -f');
    console.log('  4. git commit -m "Manual Deploy"');
    console.log('  5. git subtree push --prefix dist origin gh-pages');
    console.log('\\nAfter step 5, your site will be live!');
    console.log('To go back to coding: git checkout main');
  } else {
    console.log('✅ Deployment Complete!');
    console.log('Check your repository settings > Pages to see your live site.');
  }
});
      `.trim());

      // 5. Add Package.json (Updated to use deploy.js)
      zip.file('package.json', `
{
  "name": "app-inspector-ai",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "deploy": "npm run build && node deploy.js"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "lucide-react": "^0.344.0",
    "jszip": "^3.10.1",
    "@google/genai": "latest"
  },
  "devDependencies": {
    "@types/react": "^18.2.66",
    "@types/react-dom": "^18.2.22",
    "@types/node": "^20.0.0",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.2.2",
    "vite": "^5.2.0",
    "gh-pages": "^6.1.1"
  }
}
      `.trim());

      // 6. Add .gitignore (CRITICAL FIX)
      zip.file('.gitignore', `
node_modules
dist
.env
.DS_Store
      `.trim());

      // 7. Special handling for index.html
      try {
        const indexResponse = await fetch('index.html');
        if (indexResponse.ok) {
          let indexHtml = await indexResponse.text();
          if (!indexHtml.includes('src="/index.tsx"')) {
             indexHtml = indexHtml.replace(
               '</body>', 
               '<script type="module" src="/index.tsx"></script></body>'
             );
          }
          indexHtml = indexHtml.replace(/<script type="importmap">[\s\S]*?<\/script>/, '');
          zip.file('index.html', indexHtml);
        }
      } catch(e) { console.error("Index HTML error", e); }

      // 8. Updated README with troubleshooting
      zip.file("README.md", `# AppInspector AI

## 1. Quick Start (Local)

1. **Unzip** this folder.
2. Open terminal in this folder.
3. Run:
   \`\`\`bash
   npm install
   npm run dev
   \`\`\`

## 2. How to Host on GitHub Pages (Free)

1. **Create a Repository** on GitHub.
2. **Push your code**:
   \`\`\`bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   \`\`\`
3. **Deploy**:
   \`\`\`bash
   npm run deploy
   \`\`\`

## Troubleshooting "ENAMETOOLONG" Error
If \`npm run deploy\` fails with an error like \`spawn ENAMETOOLONG\`, it means Windows path limits are blocking the automatic script.

**Solution (Manual Deploy):**
Run these commands in your terminal one by one:
\`\`\`bash
git checkout -b gh-pages
npm run build
git add dist -f
git commit -m "Manual deploy"
git subtree push --prefix dist origin gh-pages
\`\`\`
Then switch back to your main branch:
\`\`\`bash
git checkout main
\`\`\`
`);

      const content = await zip.generateAsync({ type: "blob" });
      const url = window.URL.createObjectURL(content);
      const link = document.createElement('a');
      link.href = url;
      link.download = "AppInspector_Vite_Project.zip";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      setShowDownloadModal(true);
    } catch (e) {
      console.error("Download failed:", e);
      alert("Could not generate zip file. Please try again.");
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 relative overflow-hidden font-sans">
      {/* Background decorations */}
      <div className="absolute top-0 left-0 w-full h-96 bg-gradient-to-b from-indigo-100/50 to-transparent pointer-events-none" />
      <div className="absolute -top-20 -right-20 w-96 h-96 bg-blue-200/20 rounded-full blur-3xl pointer-events-none" />
      <div className="absolute top-40 -left-20 w-72 h-72 bg-purple-200/20 rounded-full blur-3xl pointer-events-none" />

      {/* Download Button */}
      <div className="absolute top-4 right-4 z-50">
        <button 
          onClick={downloadSourceCode}
          className="flex items-center space-x-2 bg-white/80 backdrop-blur-sm border border-indigo-100 text-indigo-700 px-4 py-2 rounded-full shadow-sm hover:shadow-md hover:bg-white transition-all text-sm font-medium"
        >
          <Download className="h-4 w-4" />
          <span className="hidden sm:inline">Download & Run</span>
        </button>
      </div>

      <main className="container mx-auto px-4 py-8 relative z-10">
        
        {/* Header */}
        {(appState === AppState.SEARCH || appState === AppState.SELECTING) && (
          <div className="text-center mb-12 animate-fade-in">
            <div className="inline-flex items-center justify-center p-3 bg-white rounded-2xl shadow-sm mb-6">
              <ShieldCheck className="h-8 w-8 text-indigo-600 mr-2" />
              <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-indigo-600 to-purple-600">
                AppInspector AI
              </span>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4 tracking-tight">
              Is that app safe?
            </h1>
            <p className="text-lg text-gray-600 max-w-2xl mx-auto">
              Search any Google Play Store app to get an instant AI-powered analysis of its reviews, authenticity, and developer background.
            </p>
          </div>
        )}

        {/* Search View */}
        {(appState === AppState.SEARCH || appState === AppState.SELECTING) && (
          <div className="animate-fade-in">
            <SearchBar onSearch={handleSearch} isLoading={isLoading && appState === AppState.SEARCH} />
            
            {error && (
              <div className="mt-6 text-center p-4 bg-red-50 text-red-600 rounded-xl max-w-lg mx-auto border border-red-100 flex items-center justify-center gap-2">
                <HelpCircle className="h-5 w-5" />
                {error}
              </div>
            )}
            
            <SearchResults results={searchResults} onSelect={handleSelectApp} />
          </div>
        )}

        {/* Loading Analysis View */}
        {appState === AppState.ANALYZING && (
          <div className="flex flex-col items-center justify-center min-h-[50vh] animate-fade-in">
            <div className="relative">
              <div className="absolute inset-0 bg-indigo-500/20 rounded-full blur-xl animate-pulse" />
              <Loader2 className="h-16 w-16 text-indigo-600 animate-spin relative z-10" />
            </div>
            <h2 className="mt-8 text-2xl font-bold text-gray-900">Analyzing {selectedApp?.name}...</h2>
            <p className="text-gray-500 mt-2 text-center max-w-md">
              Our AI agents are reading reviews, checking developer history, and verifying authenticity.
              <br/><span className="text-xs text-indigo-400 mt-2 block">(This might take up to 30s depending on AI traffic)</span>
            </p>
            
            <div className="mt-8 w-full max-w-md space-y-3">
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div className="h-full bg-indigo-600 rounded-full w-2/3 animate-[shimmer_2s_infinite]" />
              </div>
              <div className="flex justify-between text-xs text-gray-400 font-medium uppercase tracking-wider">
                <span>Collecting Data</span>
                <span>Processing</span>
                <span>Summarizing</span>
              </div>
            </div>
          </div>
        )}

        {/* Dashboard View */}
        {appState === AppState.DASHBOARD && selectedApp && analysis && (
          <Dashboard 
            app={selectedApp} 
            analysis={analysis} 
            onBack={handleBackToSearch} 
          />
        )}
      </main>

      {/* Download Instructions Modal */}
      {showDownloadModal && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/40 backdrop-blur-sm animate-fade-in">
          <div className="bg-white rounded-2xl max-w-md w-full p-6 shadow-2xl relative">
            <button 
              onClick={() => setShowDownloadModal(false)}
              className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X className="h-5 w-5" />
            </button>
            
            <div className="flex items-center justify-center w-12 h-12 bg-indigo-100 rounded-full mb-4 mx-auto">
              <Github className="h-6 w-6 text-indigo-700" />
            </div>
            
            <h3 className="text-xl font-bold text-center text-gray-900 mb-2">Project Ready!</h3>
            <p className="text-gray-500 text-center mb-6 text-sm">
              We've included files to help you host this on GitHub Pages.
            </p>
            
            <div className="space-y-4 bg-gray-50 p-4 rounded-xl border border-gray-100 text-sm">
              <div className="flex items-start">
                <span className="flex items-center justify-center w-5 h-5 bg-indigo-600 text-white rounded-full text-xs font-bold mr-3 flex-shrink-0">1</span>
                <p>Unzip and open terminal.</p>
              </div>
              <div className="flex items-start">
                <span className="flex items-center justify-center w-5 h-5 bg-indigo-600 text-white rounded-full text-xs font-bold mr-3 flex-shrink-0">2</span>
                <p>Run <code className="bg-gray-200 px-1 rounded text-indigo-700 font-mono font-bold">npm install</code>.</p>
              </div>
              <div className="flex items-start">
                <span className="flex items-center justify-center w-5 h-5 bg-indigo-600 text-white rounded-full text-xs font-bold mr-3 flex-shrink-0">3</span>
                <p>Run <code className="bg-gray-200 px-1 rounded text-indigo-700 font-mono font-bold">npm run deploy</code>.</p>
              </div>
            </div>
            
            <button 
              onClick={() => setShowDownloadModal(false)}
              className="w-full mt-6 py-3 bg-indigo-600 hover:bg-indigo-700 text-white font-medium rounded-xl transition-colors"
            >
              Got it
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;