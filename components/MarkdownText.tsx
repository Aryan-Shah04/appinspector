import React from 'react';

interface MarkdownTextProps {
  content: string;
  className?: string;
}

const MarkdownText: React.FC<MarkdownTextProps> = ({ content, className = '' }) => {
  // Guard clause for undefined or null content to prevent "cannot read properties of undefined (reading 'split')"
  if (!content) {
    return null;
  }

  const parseInline = (text: string): React.ReactNode[] => {
    // Split by **bold** or `code` syntax
    // Regex matches: (**...**) OR (`...`)
    const parts = text.split(/(\*\*.*?\*\*|`.*?`)/g);
    
    return parts.map((part, index) => {
      // Handle Bold
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={index} className="font-bold text-gray-900">{part.slice(2, -2)}</strong>;
      }
      // Handle Inline Code
      if (part.startsWith('`') && part.endsWith('`')) {
        return (
          <code key={index} className="bg-gray-100 text-indigo-700 px-1.5 py-0.5 rounded text-sm font-mono border border-gray-200">
            {part.slice(1, -1)}
          </code>
        );
      }
      // Return plain text
      return part;
    });
  };

  const lines = content.split('\n');
  const elements: React.ReactNode[] = [];
  let currentList: React.ReactNode[] = [];
  
  lines.forEach((line, i) => {
    const trimmed = line.trim();
    
    // Check for list items (bullet points)
    const isList = trimmed.startsWith('- ') || trimmed.startsWith('* ');
    
    if (isList) {
      const text = trimmed.substring(2);
      currentList.push(
        <li key={`li-${i}`} className="mb-1">
          {parseInline(text)}
        </li>
      );
    } else {
      // Flush any pending list
      if (currentList.length > 0) {
        elements.push(
          <ul key={`ul-${i}`} className="list-disc pl-5 mb-4 space-y-1 text-gray-700">
            {[...currentList]}
          </ul>
        );
        currentList = [];
      }
      
      if (!trimmed) {
        return; // Skip empty lines between paragraphs
      }
      
      // Render standard paragraph
      elements.push(
        <p key={`p-${i}`} className="mb-3 last:mb-0 leading-relaxed text-gray-700">
          {parseInline(trimmed)}
        </p>
      );
    }
  });

  // Flush remaining list at the end
  if (currentList.length > 0) {
    elements.push(
      <ul key="ul-last" className="list-disc pl-5 mb-4 space-y-1 text-gray-700">
        {[...currentList]}
      </ul>
    );
  }

  return <div className={className}>{elements}</div>;
};

export default MarkdownText;