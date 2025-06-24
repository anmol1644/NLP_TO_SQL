import { useState, useEffect } from 'react';
import { X, LayoutGrid, ArrowUpDown, AlertCircle, BarChart2, Calendar, Clock, Database, Trash2, Maximize2, Eye } from 'lucide-react';
import Visualization from './Visualization';
import InsightPanel from './InsightPanel';

interface SavedQuery {
  id: string;
  title: string;
  description?: string;
  sql: string;
  data: any[];
  timestamp: string;
  tableName?: string;
}

interface DashboardProps {
  isOpen: boolean;
  onClose: () => void;
  onAddQuery?: (query: SavedQuery) => void;
  currentData?: any[];
  currentSql?: string;
  currentTableName?: string;
  savedQueries?: SavedQuery[];
}

export default function Dashboard({ 
  isOpen, 
  onClose, 
  savedQueries = []
}: DashboardProps) {
  const [selectedQuery, setSelectedQuery] = useState<SavedQuery | null>(null);
  const [showVisualization, setShowVisualization] = useState(false);
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest'>('newest');
  const [searchTerm, setSearchTerm] = useState('');
  
  // Sort and filter queries
  const sortedQueries = [...savedQueries].sort((a, b) => {
    const dateA = new Date(a.timestamp).getTime();
    const dateB = new Date(b.timestamp).getTime();
    return sortOrder === 'newest' ? dateB - dateA : dateA - dateB;
  }).filter(query => {
    if (!searchTerm) return true;
    const searchTermLower = searchTerm.toLowerCase();
    return (
      query.title.toLowerCase().includes(searchTermLower) ||
      (query.description || '').toLowerCase().includes(searchTermLower) ||
      query.sql.toLowerCase().includes(searchTermLower)
    );
  });

  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center overflow-y-auto">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-7xl h-[90vh] flex flex-col mx-4">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl shadow-md">
              <LayoutGrid className="h-5 w-5 text-white" />
            </div>
            <h2 className="text-xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              Analytics Dashboard
            </h2>
          </div>

          <button
            onClick={onClose}
            className="p-2 text-gray-500 hover:bg-gray-100 rounded-md transition-colors"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Main content */}
        <div className="flex-1 overflow-hidden flex">
          {savedQueries.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full w-full text-center p-6">
              <div className="bg-blue-50 p-6 rounded-full mb-4">
                <AlertCircle className="h-12 w-12 text-blue-400" />
              </div>
              <h3 className="text-xl font-bold text-gray-700">No Saved Queries</h3>
              <p className="text-gray-500 mt-2 max-w-md">
                Run queries and save them to build your analytics dashboard.
              </p>
              <p className="text-gray-500 mt-4 max-w-md text-sm">
                Click the "Save to Dashboard" button on any query result to add it here.
              </p>
            </div>
          ) : (
            <>
              {/* Sidebar with saved queries list */}
              <div className="w-1/3 border-r overflow-y-auto p-4">
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-lg font-bold text-gray-800">
                      {sortedQueries.length} Saved {sortedQueries.length === 1 ? 'Query' : 'Queries'}
                    </h3>
                    <button 
                      onClick={() => setSortOrder(sortOrder === 'newest' ? 'oldest' : 'newest')}
                      className="flex items-center space-x-1 text-xs text-gray-600 hover:text-gray-800 bg-gray-100 hover:bg-gray-200 px-2 py-1 rounded-lg transition-all duration-200"
                    >
                      <ArrowUpDown className="h-3 w-3" />
                      <span>{sortOrder === 'newest' ? 'Newest First' : 'Oldest First'}</span>
                    </button>
                  </div>
                  
                  <div className="relative mb-4">
                    <input
                      type="text"
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      placeholder="Search saved queries..."
                      className="w-full px-4 py-2 pl-10 border border-gray-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <svg className="absolute left-3 top-2.5 h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                    </svg>
                  </div>
                </div>
                
                <div className="space-y-3">
                  {sortedQueries.map((query) => (
                    <div 
                      key={query.id}
                      onClick={() => setSelectedQuery(query)}
                      className={`p-3 rounded-lg border cursor-pointer transition-all duration-200 ${
                        selectedQuery?.id === query.id
                          ? 'border-blue-300 bg-blue-50 shadow-md'
                          : 'border-gray-200 hover:border-blue-200 hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h4 className="font-medium text-gray-800 truncate">{query.title}</h4>
                        <div className="flex items-center space-x-1">
                          <button 
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedQuery(query);
                              setShowVisualization(true);
                            }}
                            className="p-1 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded"
                          >
                            <BarChart2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                      
                      <div className="flex items-center text-xs text-gray-500 space-x-3 mb-2">
                        <div className="flex items-center">
                          <Calendar className="h-3 w-3 mr-1" />
                          <span>{formatDate(query.timestamp)}</span>
                        </div>
                        <div className="flex items-center">
                          <Database className="h-3 w-3 mr-1" />
                          <span>{query.data.length} rows</span>
                        </div>
                      </div>
                      
                      {query.description && (
                        <p className="text-xs text-gray-600 truncate">{query.description}</p>
                      )}
                      
                      <div className="mt-2 bg-gray-100 rounded p-2 overflow-hidden">
                        <pre className="text-xs text-gray-600 font-mono truncate">{query.sql}</pre>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Main content area */}
              <div className="flex-1 overflow-y-auto p-6">
                {selectedQuery ? (
                  <div className="space-y-6">
                    <div className="flex items-center justify-between">
                      <h3 className="text-xl font-bold text-gray-800">{selectedQuery.title}</h3>
                      <div className="flex items-center space-x-2">
                        <button 
                          onClick={() => setShowVisualization(true)}
                          className="flex items-center space-x-1 text-sm text-blue-600 hover:text-blue-800 bg-blue-100 hover:bg-blue-200 px-3 py-1.5 rounded-lg transition-all duration-200"
                        >
                          <BarChart2 className="h-4 w-4" />
                          <span>Visualize</span>
                        </button>
                      </div>
                    </div>
                    
                    {selectedQuery.description && (
                      <p className="text-gray-600 bg-gray-50 p-3 rounded-lg border border-gray-100">
                        {selectedQuery.description}
                      </p>
                    )}
                    
                    <div className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden">
                      <div className="p-4 border-b bg-gray-50">
                        <h4 className="font-medium text-gray-700">Data Preview</h4>
                      </div>
                      <div className="overflow-x-auto max-h-64">
                        <table className="min-w-full divide-y divide-gray-200">
                          <thead className="bg-gray-50">
                            <tr>
                              {selectedQuery.data[0] && Object.keys(selectedQuery.data[0]).map((header) => (
                                <th
                                  key={header}
                                  className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider sticky top-0 bg-gray-50"
                                >
                                  {header}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="bg-white divide-y divide-gray-200">
                            {selectedQuery.data.slice(0, 10).map((row, rowIndex) => (
                              <tr key={rowIndex} className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                {Object.entries(row).map(([key, value], cellIndex) => (
                                  <td key={`${rowIndex}-${cellIndex}`} className="px-4 py-2 text-sm text-gray-500">
                                    {String(value)}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      {selectedQuery.data.length > 10 && (
                        <div className="p-2 text-center text-xs text-gray-500 bg-gray-50 border-t">
                          Showing 10 of {selectedQuery.data.length} rows
                        </div>
                      )}
                    </div>
                    
                    <div className="mt-6">
                      <InsightPanel data={selectedQuery.data} query={selectedQuery.sql} />
                    </div>
                    
                    <div className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden mt-6">
                      <div className="p-4 border-b bg-gray-50">
                        <h4 className="font-medium text-gray-700">SQL Query</h4>
                      </div>
                      <div className="p-4 bg-gray-800 overflow-x-auto">
                        <pre className="text-sm text-gray-200 font-mono whitespace-pre-wrap">{selectedQuery.sql}</pre>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center">
                    <div className="bg-gray-100 p-6 rounded-full mb-4">
                      <Eye className="h-8 w-8 text-gray-400" />
                    </div>
                    <h3 className="text-lg font-medium text-gray-700">Select a query to view details</h3>
                    <p className="text-gray-500 mt-2 max-w-md">
                      Click on any saved query from the sidebar to view its details, insights, and visualizations.
                    </p>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </div>
      
      {/* Visualization Modal */}
      {showVisualization && selectedQuery && (
        <Visualization 
          data={selectedQuery.data} 
          onClose={() => setShowVisualization(false)} 
        />
      )}
    </div>
  );
} 