import React, { useState, useEffect } from 'react';
import { BarChart2, TrendingUp, AlertCircle, Info, ChevronDown, ChevronUp, Lightbulb, Zap } from 'lucide-react';

interface InsightPanelProps {
  data: any[];
  tableName?: string;
  query?: string;
  isLoading?: boolean;
}

const InsightPanel: React.FC<InsightPanelProps> = ({ data, tableName, query, isLoading = false }) => {
  const [insights, setInsights] = useState<{
    general: string[];
    statistical: { column: string; stats: { label: string; value: string | number }[] }[];
    patterns: string[];
  }>({
    general: [],
    statistical: [],
    patterns: []
  });
  
  const [expandedSections, setExpandedSections] = useState({
    general: true,
    statistical: true,
    patterns: true
  });

  useEffect(() => {
    if (data && data.length > 0) {
      // Generate insights based on the data
      const generatedInsights = generateInsights(data, query || '');
      setInsights(generatedInsights);
    }
  }, [data, query]);

  const toggleSection = (section: 'general' | 'statistical' | 'patterns') => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const generateInsights = (data: any[], query: string) => {
    const general: string[] = [];
    const statistical: { column: string; stats: { label: string; value: string | number }[] }[] = [];
    const patterns: string[] = [];
    
    // Add basic info about the data
    general.push(`This query returned ${data.length} ${data.length === 1 ? 'row' : 'rows'} of data.`);
    
    // Get column names
    const columns = Object.keys(data[0]);
    general.push(`The dataset contains ${columns.length} columns.`);
    
    // Try to identify the table type from column names or query
    const tableType = identifyTableType(columns, query);
    if (tableType) {
      general.push(`This appears to be ${tableType} data.`);
    }

    // Try to find numeric columns for basic stats
    const numericColumns = columns.filter(col => 
      typeof data[0][col] === 'number' || !isNaN(Number(data[0][col]))
    );
    
    // Date columns
    const dateColumns = columns.filter(col => 
      isDateString(data[0][col])
    );
    
    if (dateColumns.length > 0) {
      const dateCol = dateColumns[0];
      const dates = data.map(row => new Date(row[dateCol])).filter(d => !isNaN(d.getTime()));
      
      if (dates.length > 0) {
        const minDate = new Date(Math.min(...dates.map(d => d.getTime())));
        const maxDate = new Date(Math.max(...dates.map(d => d.getTime())));
        
        patterns.push(`Date range for ${dateCol}: ${formatDate(minDate)} to ${formatDate(maxDate)}`);
        
        // Check for recency
        const now = new Date();
        const daysSinceLastDate = Math.floor((now.getTime() - maxDate.getTime()) / (1000 * 60 * 60 * 24));
        if (daysSinceLastDate < 30) {
          patterns.push(`The data is recent, with the latest date only ${daysSinceLastDate} days ago.`);
        } else {
          patterns.push(`The most recent data is from ${daysSinceLastDate} days ago.`);
        }
      }
    }
    
    // Process numeric columns
    if (numericColumns.length > 0) {
      numericColumns.forEach(col => {
        try {
          const values = data.map(row => Number(row[col])).filter(val => !isNaN(val));
          if (values.length > 0) {
            const sum = values.reduce((a, b) => a + b, 0);
            const avg = sum / values.length;
            const max = Math.max(...values);
            const min = Math.min(...values);
            
            // Calculate standard deviation
            const variance = values.reduce((acc, val) => acc + Math.pow(val - avg, 2), 0) / values.length;
            const stdDev = Math.sqrt(variance);
            
            // Calculate median
            const sortedValues = [...values].sort((a, b) => a - b);
            const median = sortedValues.length % 2 === 0 
              ? (sortedValues[sortedValues.length / 2 - 1] + sortedValues[sortedValues.length / 2]) / 2
              : sortedValues[Math.floor(sortedValues.length / 2)];
            
            // Format the values appropriately
            const isDecimal = values.some(v => v % 1 !== 0);
            const formatValue = (value: number) => isDecimal ? value.toFixed(2) : value.toString();
            
            // Add to statistical insights
            statistical.push({
              column: col,
              stats: [
                { label: 'Average', value: formatValue(avg) },
                { label: 'Median', value: formatValue(median) },
                { label: 'Min', value: formatValue(min) },
                { label: 'Max', value: formatValue(max) },
                { label: 'Range', value: formatValue(max - min) },
                { label: 'Std Dev', value: formatValue(stdDev) }
              ]
            });
            
            // Look for outliers (values more than 2 std deviations from mean)
            const outliers = values.filter(val => Math.abs(val - avg) > 2 * stdDev);
            if (outliers.length > 0) {
              const percentage = ((outliers.length / values.length) * 100).toFixed(1);
              patterns.push(`${percentage}% of values in "${col}" are potential outliers (±2σ from mean).`);
            }
            
            // Check for distribution skew
            if (Math.abs(avg - median) > stdDev * 0.5) {
              patterns.push(`The distribution of "${col}" is skewed (mean vs median difference).`);
            }
          }
        } catch (error) {
          // Skip this column if there's an error
        }
      });
    }
    
    // Look for correlations between numeric columns
    if (numericColumns.length >= 2) {
      for (let i = 0; i < numericColumns.length - 1; i++) {
        for (let j = i + 1; j < numericColumns.length; j++) {
          const col1 = numericColumns[i];
          const col2 = numericColumns[j];
          
          try {
            const correlation = calculateCorrelation(data, col1, col2);
            if (Math.abs(correlation) > 0.7) {
              patterns.push(
                `Strong ${correlation > 0 ? 'positive' : 'negative'} correlation (${correlation.toFixed(2)}) between "${col1}" and "${col2}".`
              );
            }
          } catch (error) {
            // Skip if error
          }
        }
      }
    }
    
    return { general, statistical, patterns };
  };
  
  // Helper function to identify table type
  const identifyTableType = (columns: string[], query: string): string | null => {
    const columnsLower = columns.map(c => c.toLowerCase());
    const queryLower = query.toLowerCase();
    
    if (columnsLower.includes('salesorderid') || columnsLower.includes('orderid') || 
        columnsLower.includes('orderdate') || queryLower.includes('sales')) {
      return 'sales order';
    }
    
    if (columnsLower.includes('customerid') || columnsLower.includes('customer') || 
        queryLower.includes('customer')) {
      return 'customer';
    }
    
    if (columnsLower.includes('productid') || columnsLower.includes('product') || 
        queryLower.includes('product')) {
      return 'product';
    }
    
    if (columnsLower.includes('employeeid') || columnsLower.includes('employee') || 
        queryLower.includes('employee')) {
      return 'employee';
    }
    
    return null;
  };
  
  // Helper function to check if a string is a date
  const isDateString = (value: any): boolean => {
    if (typeof value !== 'string') return false;
    const date = new Date(value);
    return !isNaN(date.getTime());
  };
  
  // Helper function to format date
  const formatDate = (date: Date): string => {
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };
  
  // Helper function to calculate correlation
  const calculateCorrelation = (data: any[], col1: string, col2: string): number => {
    const pairs = data
      .map(row => [Number(row[col1]), Number(row[col2])])
      .filter(pair => !isNaN(pair[0]) && !isNaN(pair[1]));
    
    if (pairs.length < 3) return 0;
    
    const n = pairs.length;
    
    // Calculate means
    const mean1 = pairs.reduce((sum, pair) => sum + pair[0], 0) / n;
    const mean2 = pairs.reduce((sum, pair) => sum + pair[1], 0) / n;
    
    // Calculate variances and covariance
    let variance1 = 0;
    let variance2 = 0;
    let covariance = 0;
    
    for (const pair of pairs) {
      const diff1 = pair[0] - mean1;
      const diff2 = pair[1] - mean2;
      
      variance1 += diff1 * diff1;
      variance2 += diff2 * diff2;
      covariance += diff1 * diff2;
    }
    
    variance1 /= n;
    variance2 /= n;
    covariance /= n;
    
    // Calculate correlation
    const stdDev1 = Math.sqrt(variance1);
    const stdDev2 = Math.sqrt(variance2);
    
    if (stdDev1 === 0 || stdDev2 === 0) return 0;
    
    return covariance / (stdDev1 * stdDev2);
  };

  if (isLoading) {
    return (
      <div className="bg-white p-6 rounded-xl shadow-lg border border-indigo-100 animate-pulse">
        <div className="flex items-center mb-6">
          <div className="w-6 h-6 bg-indigo-200 rounded-md mr-3"></div>
          <div className="h-6 bg-indigo-200 rounded w-1/3"></div>
        </div>
        <div className="space-y-4">
          <div className="h-4 bg-gray-200 rounded w-3/4"></div>
          <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          <div className="h-4 bg-gray-200 rounded w-5/6"></div>
          <div className="grid grid-cols-2 gap-4 mt-6">
            <div className="h-24 bg-gray-100 rounded"></div>
            <div className="h-24 bg-gray-100 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  const hasInsights = insights.general.length > 0 || insights.statistical.length > 0 || insights.patterns.length > 0;

  return (
    <div className="bg-gradient-to-br from-white to-indigo-50 p-6 rounded-xl shadow-lg border border-indigo-100 transition-all duration-300">
      <div className="flex items-center mb-6">
        <div className="p-2 bg-indigo-100 rounded-lg mr-3">
          <Lightbulb className="h-5 w-5 text-indigo-600" />
        </div>
        <h3 className="text-lg font-semibold bg-gradient-to-r from-indigo-700 to-purple-700 bg-clip-text text-transparent">
          Data Insights
        </h3>
      </div>
      
      {hasInsights ? (
        <div className="space-y-5">
          {/* General Insights */}
          {insights.general.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden transition-all duration-300">
              <button 
                onClick={() => toggleSection('general')}
                className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center">
                  <div className="p-1.5 bg-blue-100 rounded-md mr-3">
                    <Info className="h-4 w-4 text-blue-600" />
                  </div>
                  <h4 className="font-medium text-gray-800">General Information</h4>
                </div>
                {expandedSections.general ? (
                  <ChevronUp className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                )}
              </button>
              
              {expandedSections.general && (
                <div className="p-4 pt-0 bg-gradient-to-br from-blue-50/30 to-white animate-in slide-in-from-top-2 duration-300">
                  <ul className="space-y-2 pl-10 list-disc text-sm text-gray-600">
                    {insights.general.map((insight, index) => (
                      <li key={`general-${index}`} className="animate-in fade-in duration-300" style={{ animationDelay: `${index * 100}ms` }}>
                        {insight}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
          
          {/* Statistical Insights */}
          {insights.statistical.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden transition-all duration-300">
              <button 
                onClick={() => toggleSection('statistical')}
                className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center">
                  <div className="p-1.5 bg-emerald-100 rounded-md mr-3">
                    <BarChart2 className="h-4 w-4 text-emerald-600" />
                  </div>
                  <h4 className="font-medium text-gray-800">Statistical Analysis</h4>
                </div>
                {expandedSections.statistical ? (
                  <ChevronUp className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                )}
              </button>
              
              {expandedSections.statistical && (
                <div className="p-4 pt-2 bg-gradient-to-br from-emerald-50/30 to-white animate-in slide-in-from-top-2 duration-300">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {insights.statistical.map((column, colIndex) => (
                      <div 
                        key={`stat-${colIndex}`} 
                        className="bg-white p-3 rounded-lg border border-emerald-100 shadow-sm hover:shadow-md transition-all duration-300 animate-in fade-in"
                        style={{ animationDelay: `${colIndex * 100}ms` }}
                      >
                        <h5 className="text-sm font-semibold text-gray-700 mb-2 pb-1 border-b border-emerald-100 flex items-center">
                          <TrendingUp className="h-3.5 w-3.5 text-emerald-500 mr-1.5" />
                          {column.column.replace(/_/g, ' ')}
                        </h5>
                        <div className="grid grid-cols-3 gap-y-2 gap-x-3">
                          {column.stats.map((stat, statIndex) => (
                            <div key={`stat-${colIndex}-${statIndex}`} className="flex flex-col">
                              <span className="text-xs text-gray-500">{stat.label}:</span>
                              <span className="text-sm font-medium text-gray-800">{stat.value}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
          
          {/* Pattern Insights */}
          {insights.patterns.length > 0 && (
            <div className="bg-white rounded-lg shadow-sm border border-gray-100 overflow-hidden transition-all duration-300">
              <button 
                onClick={() => toggleSection('patterns')}
                className="w-full flex items-center justify-between p-4 text-left hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-center">
                  <div className="p-1.5 bg-amber-100 rounded-md mr-3">
                    <Zap className="h-4 w-4 text-amber-600" />
                  </div>
                  <h4 className="font-medium text-gray-800">Patterns & Observations</h4>
                </div>
                {expandedSections.patterns ? (
                  <ChevronUp className="h-4 w-4 text-gray-500" />
                ) : (
                  <ChevronDown className="h-4 w-4 text-gray-500" />
                )}
              </button>
              
              {expandedSections.patterns && (
                <div className="p-4 pt-0 bg-gradient-to-br from-amber-50/30 to-white animate-in slide-in-from-top-2 duration-300">
                  <ul className="space-y-2 pl-10 list-disc text-sm text-gray-600">
                    {insights.patterns.map((pattern, index) => (
                      <li key={`pattern-${index}`} className="animate-in fade-in duration-300" style={{ animationDelay: `${index * 100}ms` }}>
                        {pattern}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        <div className="bg-white p-8 rounded-lg shadow-sm border border-gray-100 flex flex-col items-center justify-center text-center">
          <AlertCircle className="h-10 w-10 text-gray-300 mb-3" />
          <p className="text-gray-500">No insights available for this dataset.</p>
        </div>
      )}
    </div>
  );
};

export default InsightPanel; 