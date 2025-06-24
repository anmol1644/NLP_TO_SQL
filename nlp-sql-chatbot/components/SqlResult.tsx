import { AlertTriangle, CheckCircle, Database, Copy, Download } from 'lucide-react';
import { useState } from 'react';

interface SqlResultProps {
  sql: string;
  data?: any[];
  error?: string;
}

export default function SqlResult({ sql, data, error }: SqlResultProps) {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  if (error) {
    return (
      <div className="border border-red-200 rounded-2xl p-6 bg-gradient-to-r from-red-50 to-rose-50 shadow-lg animate-in slide-in-from-bottom-2 duration-500">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 bg-red-100 rounded-xl">
            <AlertTriangle className="h-5 w-5 text-red-600" />
          </div>
          <h3 className="text-lg font-bold text-red-800">Query Error</h3>
        </div>
        
        <div className="bg-red-100/50 rounded-xl p-4 mb-4">
          <pre className="text-sm text-red-700 whitespace-pre-wrap font-mono leading-relaxed">{error}</pre>
        </div>
        
        <div className="pt-4 border-t border-red-200">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-semibold text-red-800 flex items-center space-x-2">
              <Database className="h-4 w-4" />
              <span>SQL Query</span>
            </h4>
            <button
              onClick={() => copyToClipboard(sql)}
              className="flex items-center space-x-1 text-xs text-red-600 hover:text-red-800 bg-red-100 hover:bg-red-200 px-3 py-1 rounded-lg transition-all duration-200"
            >
              <Copy className="h-3 w-3" />
              <span>{copied ? 'Copied!' : 'Copy'}</span>
            </button>
          </div>
          <div className="bg-red-100 rounded-xl p-3 overflow-x-auto">
            <pre className="text-sm text-red-700 font-mono">{sql}</pre>
          </div>
        </div>
      </div>
    );
  }

  if (!data || !data.length) {
    return (
      <div className="border border-amber-200 rounded-2xl p-6 bg-gradient-to-r from-amber-50 to-yellow-50 shadow-lg animate-in slide-in-from-bottom-2 duration-500">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 bg-amber-100 rounded-xl">
            <CheckCircle className="h-5 w-5 text-amber-600" />
          </div>
          <h3 className="text-lg font-bold text-amber-800">Query Executed</h3>
        </div>
        
        <p className="text-sm text-amber-700 mb-4 bg-amber-100/50 p-3 rounded-xl">
          The query executed successfully but returned no data.
        </p>
        
        <div className="pt-4 border-t border-amber-200">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-semibold text-amber-800 flex items-center space-x-2">
              <Database className="h-4 w-4" />
              <span>SQL Query</span>
            </h4>
            <button
              onClick={() => copyToClipboard(sql)}
              className="flex items-center space-x-1 text-xs text-amber-600 hover:text-amber-800 bg-amber-100 hover:bg-amber-200 px-3 py-1 rounded-lg transition-all duration-200"
            >
              <Copy className="h-3 w-3" />
              <span>{copied ? 'Copied!' : 'Copy'}</span>
            </button>
          </div>
          <div className="bg-amber-100 rounded-xl p-3 overflow-x-auto">
            <pre className="text-sm text-amber-700 font-mono">{sql}</pre>
          </div>
        </div>
      </div>
    );
  }

  const headers = Object.keys(data[0]);

  return (
    <div className="border border-emerald-200 rounded-2xl p-6 bg-gradient-to-r from-emerald-50 to-green-50 shadow-lg animate-in slide-in-from-bottom-2 duration-500">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-emerald-100 rounded-xl">
            <CheckCircle className="h-5 w-5 text-emerald-600" />
          </div>
          <h3 className="text-lg font-bold text-emerald-800">Query Results</h3>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="text-xs bg-emerald-100 text-emerald-800 px-3 py-1 rounded-full font-semibold">
            {data.length} {data.length === 1 ? 'row' : 'rows'}
          </span>
          <button
            onClick={() => copyToClipboard(JSON.stringify(data, null, 2))}
            className="flex items-center space-x-1 text-xs text-emerald-600 hover:text-emerald-800 bg-emerald-100 hover:bg-emerald-200 px-3 py-1 rounded-lg transition-all duration-200"
          >
            <Download className="h-3 w-3" />
            <span>Export</span>
          </button>
        </div>
      </div>
      
      {/* Data Table */}
      <div className="mb-6 overflow-hidden rounded-xl border border-emerald-200 shadow-inner">
        <div className="overflow-x-auto max-h-96">
          <table className="min-w-full">
            <thead>
              <tr className="bg-gradient-to-r from-emerald-100 to-green-100">
                {headers.map((header) => (
                  <th
                    key={header}
                    className="px-4 py-3 text-left text-xs font-bold text-emerald-800 uppercase tracking-wider border-b border-emerald-200 sticky top-0 bg-emerald-100"
                  >
                    {header.replace(/_/g, ' ')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-emerald-100">
              {data.map((row, rowIndex) => (
                <tr 
                  key={rowIndex} 
                  className={`transition-colors duration-200 hover:bg-emerald-50/50 ${
                    rowIndex % 2 === 0 ? 'bg-white' : 'bg-emerald-50/20'
                  }`}
                >
                  {headers.map((header) => (
                    <td 
                      key={`${rowIndex}-${header}`} 
                      className="px-4 py-3 text-sm text-gray-700 border-r border-emerald-100 last:border-r-0 font-medium"
                    >
                      {row[header] !== null ? (
                        <span className="block truncate max-w-xs" title={String(row[header])}>
                          {String(row[header])}
                        </span>
                      ) : (
                        <span className="text-gray-400 italic font-normal">NULL</span>
                      )}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      
      {/* SQL Query Section */}
      <div className="pt-4 border-t border-emerald-200">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold text-emerald-800 flex items-center space-x-2">
            <Database className="h-4 w-4" />
            <span>SQL Query</span>
          </h4>
          <button
            onClick={() => copyToClipboard(sql)}
            className="flex items-center space-x-1 text-xs text-emerald-600 hover:text-emerald-800 bg-emerald-100 hover:bg-emerald-200 px-3 py-1 rounded-lg transition-all duration-200"
          >
            <Copy className="h-3 w-3" />
            <span>{copied ? 'Copied!' : 'Copy'}</span>
          </button>
        </div>
        <div className="bg-emerald-100 rounded-xl p-4 overflow-x-auto">
          <pre className="text-sm text-emerald-700 font-mono leading-relaxed">{sql}</pre>
        </div>
      </div>
    </div>
  );
}