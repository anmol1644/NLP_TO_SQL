import { AlertTriangle, CheckCircle, Database, Copy, Download, ChevronLeft, ChevronRight, BarChart, Lightbulb, BookmarkPlus } from 'lucide-react';
import { useState } from 'react';
import Visualization from './Visualization';
import InsightPanel from './InsightPanel';

interface PaginationInfo {
  table_id: string;
  current_page: number;
  total_pages: number;
  total_rows: number;
  page_size: number;
  has_next?: boolean;
  has_prev?: boolean;
}

interface SqlResultProps {
  sql: string;
  data?: any[];
  error?: string;
  title?: string;
  description?: string;
  pagination?: PaginationInfo;
  onPageChange?: (page: number) => void;
  sessionId?: string;
  tableId?: string;
  onSaveToAnalytics?: (query: any) => void;
}

export default function SqlResult({ 
  sql, 
  data, 
  error, 
  title, 
  description,
  pagination,
  onPageChange,
  sessionId,
  tableId,
  onSaveToAnalytics
}: SqlResultProps) {
  const [copied, setCopied] = useState(false);
  const [showVisualization, setShowVisualization] = useState(false);
  const [showInsights, setShowInsights] = useState(true);
  const [isSaved, setIsSaved] = useState(false);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handlePageChange = (newPage: number) => {
    if (onPageChange && (pagination || tableId)) {
      // Use the tableId from props if available, otherwise from pagination
      const effectiveTableId = tableId || (pagination ? pagination.table_id : null);
      
      if (effectiveTableId) {
        console.log(`SqlResult requesting page ${newPage} for table ${effectiveTableId}`);
        onPageChange(newPage);
      } else {
        console.error('Cannot change page: missing table ID');
      }
    }
  };

  const handleSaveToAnalytics = () => {
    if (onSaveToAnalytics && data) {
      const savedQuery = {
        id: `query-${Date.now()}`,
        title: title || "Saved Query",
        description: description || "",
        sql,
        data,
        timestamp: new Date().toISOString(),
        tableName: tableId
      };
      
      onSaveToAnalytics(savedQuery);
      setIsSaved(true);
      
      // Reset saved status after a while
      setTimeout(() => setIsSaved(false), 3000);
    }
  };

  if (error) {
    return (
      <div className="border border-red-200 rounded-2xl p-6 bg-gradient-to-r from-red-50 to-rose-50 shadow-lg animate-in slide-in-from-bottom-2 duration-500">
        <div className="flex items-center space-x-3 mb-4">
          <div className="p-2 bg-red-100 rounded-xl">
            <AlertTriangle className="h-5 w-5 text-red-600" />
          </div>
          <h3 className="text-lg font-bold text-red-800">{title || "Query Error"}</h3>
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
          <h3 className="text-lg font-bold text-amber-800">{title || "Query Executed"}</h3>
        </div>
        
        {description && (
          <p className="text-sm text-amber-700 mb-4">{description}</p>
        )}
        
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

  // Safely get headers - make sure data[0] exists
  const headers = data[0] ? Object.keys(data[0]) : [];

  return (
    <div className="border border-emerald-200 rounded-2xl p-6 bg-gradient-to-r from-emerald-50 to-green-50 shadow-lg animate-in slide-in-from-bottom-2 duration-500">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-emerald-100 rounded-xl">
            <CheckCircle className="h-5 w-5 text-emerald-600" />
          </div>
          <h3 className="text-lg font-bold text-emerald-800">{title || "Query Results"}</h3>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className="text-xs bg-emerald-100 text-emerald-800 px-3 py-1 rounded-full font-semibold">
            {pagination ? `${data?.length || 0} of ${pagination.total_rows}` : `${data?.length || 0}`} {data?.length === 1 ? 'row' : 'rows'}
          </span>
          <button
            onClick={() => copyToClipboard(JSON.stringify(data, null, 2))}
            className="flex items-center space-x-1 text-xs text-emerald-600 hover:text-emerald-800 bg-emerald-100 hover:bg-emerald-200 px-3 py-1 rounded-lg transition-all duration-200"
          >
            <Download className="h-3 w-3" />
            <span>Export</span>
          </button>
          <button
            onClick={() => setShowVisualization(true)}
            className="flex items-center space-x-1 text-xs text-blue-600 hover:text-blue-800 bg-blue-100 hover:bg-blue-200 px-3 py-1 rounded-lg transition-all duration-200"
          >
            <BarChart className="h-3 w-3" />
            <span>Visualize</span>
          </button>
          <button
            onClick={() => setShowInsights(!showInsights)}
            className={`flex items-center space-x-1 text-xs ${
              showInsights ? 'text-indigo-600 hover:text-indigo-800 bg-indigo-100 hover:bg-indigo-200' : 'text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200'
            } px-3 py-1 rounded-lg transition-all duration-200`}
          >
            <Lightbulb className="h-3 w-3" />
            <span>{showInsights ? 'Hide Insights' : 'Show Insights'}</span>
          </button>
          {onSaveToAnalytics && (
            <button
              onClick={handleSaveToAnalytics}
              disabled={isSaved}
              className={`flex items-center space-x-1 text-xs ${
                isSaved 
                  ? 'text-purple-800 bg-purple-200 cursor-default' 
                  : 'text-purple-600 hover:text-purple-800 bg-purple-100 hover:bg-purple-200'
              } px-3 py-1 rounded-lg transition-all duration-200`}
            >
              <BookmarkPlus className="h-3 w-3" />
              <span>{isSaved ? 'Saved!' : 'Save to Dashboard'}</span>
            </button>
          )}
        </div>
      </div>
      
      {description && (
        <p className="text-sm text-emerald-700 mb-4 bg-emerald-100/50 p-3 rounded-xl">
          {description}
        </p>
      )}
      
      {/* Data Table */}
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
            {data?.map((row, rowIndex) => (
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
                    {row[header] !== null && row[header] !== undefined ? (
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
      
      {/* Pagination Controls */}
      {pagination && pagination.total_pages > 1 && (
        <div className="flex justify-between items-center mt-4">
          <div className="text-sm text-gray-600">
            Page {pagination.current_page} of {pagination.total_pages}
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => handlePageChange(pagination.current_page - 1)}
              disabled={pagination.current_page <= 1}
              className={`flex items-center space-x-1 px-3 py-2 rounded-lg text-sm font-medium ${
                pagination.current_page > 1
                  ? 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200' 
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              }`}
            >
              <ChevronLeft className="h-4 w-4" />
              <span>Previous</span>
            </button>
            <button
              onClick={() => handlePageChange(pagination.current_page + 1)}
              disabled={pagination.current_page >= pagination.total_pages}
              className={`flex items-center space-x-1 px-3 py-2 rounded-lg text-sm font-medium ${
                pagination.current_page < pagination.total_pages
                  ? 'bg-emerald-100 text-emerald-700 hover:bg-emerald-200' 
                  : 'bg-gray-100 text-gray-400 cursor-not-allowed'
              }`}
            >
              <span>Next</span>
              <ChevronRight className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
      
      {/* Insights Panel */}
      {showInsights && data && data.length > 0 && (
        <div className="mt-6">
          <InsightPanel data={data} tableName={tableId} query={sql} />
        </div>
      )}
      
      {/* SQL Query Section */}
      <div className="mt-6 pt-4 border-t border-emerald-200">
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
      
      {/* Visualization Modal */}
      {showVisualization && (
        <Visualization data={data} onClose={() => setShowVisualization(false)} />
      )}
    </div>
  );
}