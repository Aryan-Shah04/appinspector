import React from 'react';
import { AppSearchResult } from '../types';
import { Smartphone, Star, ArrowRight } from 'lucide-react';

interface SearchResultsProps {
  results: AppSearchResult[];
  onSelect: (app: AppSearchResult) => void;
}

const SearchResults: React.FC<SearchResultsProps> = ({ results, onSelect }) => {
  if (results.length === 0) return null;

  return (
    <div className="w-full max-w-4xl mx-auto mt-12 animate-fade-in">
      <h3 className="text-xl font-semibold text-gray-800 mb-6">Found Applications</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map((app, index) => (
          <div 
            key={index}
            onClick={() => onSelect(app)}
            className="bg-white rounded-2xl p-6 shadow-sm hover:shadow-xl border border-gray-100 hover:border-indigo-100 cursor-pointer transition-all duration-300 group flex flex-col h-full"
          >
            <div className="flex items-start justify-between mb-4">
              <div className="bg-indigo-50 p-3 rounded-xl">
                <Smartphone className="h-6 w-6 text-indigo-600" />
              </div>
              {app.rating && app.rating !== 'N/A' && (
                <div className="flex items-center space-x-1 bg-yellow-50 px-2 py-1 rounded-lg">
                  <Star className="h-3 w-3 text-yellow-500 fill-current" />
                  <span className="text-xs font-bold text-yellow-700">{app.rating}</span>
                </div>
              )}
            </div>
            
            <h4 className="text-lg font-bold text-gray-900 mb-1 line-clamp-1 group-hover:text-indigo-600 transition-colors">
              {app.name}
            </h4>
            <p className="text-sm text-gray-500 mb-3 font-medium">{app.developer}</p>
            <p className="text-sm text-gray-600 line-clamp-3 mb-4 flex-grow">
              {app.description}
            </p>
            
            <div className="flex items-center text-indigo-600 text-sm font-semibold mt-auto pt-4 border-t border-gray-50">
              Analyze App <ArrowRight className="h-4 w-4 ml-1 group-hover:translate-x-1 transition-transform" />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default SearchResults;