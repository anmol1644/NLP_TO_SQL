import { useState, useEffect } from 'react';
import { 
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, 
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, AreaChart, Area, ComposedChart, 
  RadialBarChart, RadialBar, Treemap
} from 'recharts';
import { 
  BarChart2, LineChart as LineChartIcon, PieChart as PieChartIcon, 
  ScatterChart as ScatterIcon, AreaChart as AreaIcon, Activity,
  CircleUser as RadialIcon, Triangle, X, Check, Settings, ChevronDown
} from 'lucide-react';

// Define chart types
export type ChartType = 'bar' | 'line' | 'pie' | 'scatter' | 'area' | 'composed' | 'radial' | 'treemap';

interface VisualizationProps {
  data: any[];
  onClose: () => void;
  embedded?: boolean;
}

interface ChartDataItem {
  name: string;
  value: number;
}

interface ChartOption {
  type: ChartType;
  label: string;
  icon: React.ReactNode;
  recommended?: boolean;
  compatible: boolean;
  description?: string;
}

export default function Visualization({ data, onClose, embedded = false }: VisualizationProps) {
  const [chartType, setChartType] = useState<ChartType>('bar');
  const [xAxis, setXAxis] = useState<string>('');
  const [yAxis, setYAxis] = useState<string>('');
  const [secondaryYAxis, setSecondaryYAxis] = useState<string>('');
  const [availableColumns, setAvailableColumns] = useState<string[]>([]);
  const [numericalColumns, setNumericalColumns] = useState<string[]>([]);
  const [categoricalColumns, setCategoricalColumns] = useState<string[]>([]);
  const [recommendedChart, setRecommendedChart] = useState<ChartType>('bar');
  const [showOptions, setShowOptions] = useState(false);
  const [autoMode, setAutoMode] = useState(true);
  
  // COLORS for charts
  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#d53e4f', '#66c2a5', '#fc8d62'];
  
  // Detect column types and recommend chart when data changes
  useEffect(() => {
    if (!data || data.length === 0) return;
    
    const columns = Object.keys(data[0]);
    setAvailableColumns(columns);
    
    // Identify numerical and categorical columns
    const numerical: string[] = [];
    const categorical: string[] = [];
    const dateColumns: string[] = [];
    
    columns.forEach(col => {
      // Check if the column has numeric values
      const isNumeric = data.every(row => {
        const val = row[col];
        return val === null || val === undefined || !isNaN(Number(val));
      });

      // Check if the column might contain dates
      const mightBeDate = data.every(row => {
        const val = row[col];
        if (val === null || val === undefined) return true;
        const dateVal = new Date(val);
        return !isNaN(dateVal.getTime());
      });
      
      if (isNumeric) {
        numerical.push(col);
      } else {
        categorical.push(col);
      }

      if (mightBeDate) {
        dateColumns.push(col);
      }
    });
    
    setNumericalColumns(numerical);
    setCategoricalColumns(categorical);
    
    // Recommend chart type based on data characteristics
    let recommendedType: ChartType = 'bar';
    
    // If we have time/date data, recommend a line chart
    if (dateColumns.length > 0 && numerical.length > 0) {
      recommendedType = 'line';
      // For date column as X, set it as default
      setXAxis(dateColumns[0]);
    } 
    // If we have few categorical items with numeric values, pie chart is often good
    else if (categorical.length > 0 && numerical.length > 0) {
      const uniqueCategories = new Set();
      data.forEach(row => uniqueCategories.add(row[categorical[0]]));
      
      if (uniqueCategories.size <= 8) {
        recommendedType = 'pie';
        setXAxis(categorical[0]);
      } else {
        recommendedType = 'bar';
        setXAxis(categorical[0]);
      }
    } 
    // If we have multiple numerical columns, scatter plot might be interesting
    else if (numerical.length >= 2) {
      recommendedType = 'scatter';
      setXAxis(numerical[0]);
    } 
    // Default to bar chart
    else {
      recommendedType = 'bar';
      if (categorical.length > 0) {
        setXAxis(categorical[0]);
      } else if (columns.length > 0) {
        setXAxis(columns[0]);
      }
    }
    
    // Set the recommended chart type
    setRecommendedChart(recommendedType);
    setChartType(recommendedType);
    
    // Set default axes based on data types
    // X-axis: prefer categorical or date
    if (categorical.length > 0) {
      setXAxis(categorical[0]);
    } else if (dateColumns.length > 0) {
      setXAxis(dateColumns[0]);
    } else if (columns.length > 0) {
      setXAxis(columns[0]);
    }
    
    // Y-axis: prefer numerical
    if (numerical.length > 0) {
      setYAxis(numerical[0]);
      // If we have at least 2 numerical columns, set second Y for composed chart
      if (numerical.length > 1) {
        setSecondaryYAxis(numerical[1]);
      }
    } else if (columns.length > 1) {
      setYAxis(columns[1]);
    } else if (columns.length > 0) {
      setYAxis(columns[0]);
    }
  }, [data]);
  
  // Determine which chart types are compatible with the current data
  const getChartOptions = (): ChartOption[] => {
    const hasNumericalX = numericalColumns.includes(xAxis);
    const hasNumericalY = numericalColumns.includes(yAxis);
    const hasCategoricalX = categoricalColumns.includes(xAxis);
    
    // Number of unique values in X axis (for pie charts)
    let xAxisUniqueValues = new Set();
    if (data && data.length > 0) {
      data.forEach(row => xAxisUniqueValues.add(row[xAxis]));
    }
    
    return [
      {
        type: 'bar',
        label: 'Bar Chart',
        icon: <BarChart2 className="h-4 w-4" />,
        compatible: true, // Bar charts work with most data
        recommended: recommendedChart === 'bar',
        description: 'Best for comparing values across categories'
      },
      {
        type: 'line',
        label: 'Line Chart',
        icon: <LineChartIcon className="h-4 w-4" />,
        compatible: true, // Line charts also work with most data
        recommended: recommendedChart === 'line',
        description: 'Best for showing trends over time or continuous data'
      },
      {
        type: 'area',
        label: 'Area Chart',
        icon: <AreaIcon className="h-4 w-4" />,
        compatible: true,
        recommended: recommendedChart === 'area',
        description: 'Similar to line chart but with filled areas'
      },
      {
        type: 'pie',
        label: 'Pie Chart',
        icon: <PieChartIcon className="h-4 w-4" />,
        compatible: xAxisUniqueValues.size <= 10 && hasCategoricalX && hasNumericalY,
        recommended: recommendedChart === 'pie',
        description: 'Best for showing proportions of a whole (limited to 10 categories)'
      },
      {
        type: 'scatter',
        label: 'Scatter Plot',
        icon: <ScatterIcon className="h-4 w-4" />,
        compatible: hasNumericalX && hasNumericalY,
        recommended: recommendedChart === 'scatter',
        description: 'Best for showing correlation between two numeric variables'
      },
      {
        type: 'composed',
        label: 'Composed Chart',
        icon: <Activity className="h-4 w-4" />,
        compatible: numericalColumns.length >= 2,
        recommended: recommendedChart === 'composed',
        description: 'Combines multiple chart types (bar and line)'
      },
      {
        type: 'radial',
        label: 'Radial Bar',
        icon: <RadialIcon className="h-4 w-4" />,
        compatible: hasNumericalY && xAxisUniqueValues.size <= 8,
        recommended: recommendedChart === 'radial',
        description: 'Circular bar chart good for part-to-whole comparisons'
      },
      {
        type: 'treemap',
        label: 'Treemap',
        icon: <Triangle className="h-4 w-4" />,
        compatible: hasNumericalY,
        recommended: recommendedChart === 'treemap',
        description: 'Hierarchical data where areas represent values'
      }
    ];
  };
  
  const chartOptions = getChartOptions();
  const compatibleCharts = chartOptions.filter(option => option.compatible);
  
  // Prepare data for visualization
  const prepareChartData = (): any[] => {
    if (!xAxis || !yAxis || !data || data.length === 0) return [];

    // For pie charts, we need to aggregate data by xAxis
    if (chartType === 'pie' || chartType === 'radial' || chartType === 'treemap') {
      const aggregated: { [key: string]: number } = {};
      data.forEach(row => {
        const key = String(row[xAxis] || 'Unknown');
        const value = Number(row[yAxis] || 0);
        
        if (aggregated[key]) {
          aggregated[key] += value;
        } else {
          aggregated[key] = value;
        }
      });
      
      return Object.entries(aggregated).map(([name, value]) => ({ name, value }));
    }
    
    // For scatter plots
    if (chartType === 'scatter') {
      return data.map(row => ({
        name: row[xAxis] || 'Unknown',
        x: Number(row[xAxis] || 0),
        y: Number(row[yAxis] || 0)
      }));
    }

    // For composed charts with two y-axes
    if (chartType === 'composed' && secondaryYAxis) {
      return data.map(row => ({
        name: row[xAxis] || 'Unknown',
        primary: Number(row[yAxis] || 0),
        secondary: Number(row[secondaryYAxis] || 0)
      }));
    }
    
    // For bar, line, and area charts
    return data.map(row => ({
      name: row[xAxis] || 'Unknown',
      value: Number(row[yAxis] || 0)
    }));
  };
  
  const chartData = prepareChartData();

  const renderAutoChart = () => {
    const chartToRender = autoMode ? recommendedChart : chartType;
    
    switch (chartToRender) {
      case 'bar':
        return (
          <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" name={yAxis} fill="#8884d8" />
          </BarChart>
        );
      
      case 'line':
        return (
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="value" name={yAxis} stroke="#8884d8" activeDot={{ r: 8 }} />
          </LineChart>
        );
      
      case 'area':
        return (
          <AreaChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area type="monotone" dataKey="value" name={yAxis} fill="#8884d8" stroke="#8884d8" />
          </AreaChart>
        );
      
      case 'pie':
        return (
          <PieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              labelLine={true}
              label={({name, percent}: {name: string, percent: number}) => `${name}: ${(percent * 100).toFixed(0)}%`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number) => value} />
            <Legend />
          </PieChart>
        );
      
      case 'scatter':
        return (
          <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" name={xAxis} type="number" />
            <YAxis dataKey="y" name={yAxis} type="number" />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Legend />
            <Scatter name={`${xAxis} vs ${yAxis}`} data={chartData} fill="#8884d8" />
          </ScatterChart>
        );
      
      case 'composed':
        return (
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 30 }}>
            <CartesianGrid stroke="#f5f5f5" />
            <XAxis dataKey="name" />
            <YAxis yAxisId="left" />
            <YAxis yAxisId="right" orientation="right" />
            <Tooltip />
            <Legend />
            <Bar yAxisId="left" dataKey="primary" name={yAxis} fill="#8884d8" />
            <Line yAxisId="right" type="monotone" dataKey="secondary" name={secondaryYAxis} stroke="#ff7300" />
          </ComposedChart>
        );
      
      case 'radial':
        return (
          <RadialBarChart 
            cx="50%" 
            cy="50%" 
            innerRadius="10%" 
            outerRadius="80%" 
            barSize={10} 
            data={chartData}
          >
            <RadialBar
              label={{ position: 'insideStart', fill: '#888' }}
              background
              dataKey="value"
            />
            <Legend iconSize={10} layout="vertical" verticalAlign="middle" align="right" />
            <Tooltip />
          </RadialBarChart>
        );
      
      case 'treemap':
        return (
          <Treemap
            width={400}
            height={200}
            data={chartData}
            dataKey="value"
            nameKey="name"
            aspectRatio={4/3}
            stroke="#fff"
            fill="#8884d8"
          >
            {
              chartData.map((entry, index) => (
                <Cell 
                  key={`cell-${index}`}
                  fill={COLORS[index % COLORS.length]}
                />
              ))
            }
          </Treemap>
        );
      
      default:
        return (
          <div className="text-gray-500 text-center">
            <p>Select valid axes to display chart</p>
          </div>
        );
    }
  };
  
  // If embedded in dashboard, render a simplified version
  if (embedded) {
    return (
      <div className="h-full w-full">
        {chartData.length > 0 ? (
          <ResponsiveContainer width="100%" height="100%">
            {renderAutoChart()}
          </ResponsiveContainer>
        ) : (
          <div className="text-gray-500 text-center h-full flex items-center justify-center">
            <p>No data available to visualize</p>
          </div>
        )}
      </div>
    );
  }
  
  // Full visualization modal
  return (
    <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm flex items-center justify-center overflow-y-auto p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-5xl border border-blue-100 animate-in zoom-in-95 duration-300">
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-bold text-gray-800">Visualize Data</h2>
            <div className="flex items-center space-x-3">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="auto-mode"
                  checked={autoMode}
                  onChange={(e) => setAutoMode(e.target.checked)}
                  className="mr-2 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                <label htmlFor="auto-mode" className="text-sm text-gray-600">Auto-recommend chart type</label>
              </div>
              <button 
                onClick={onClose}
                className="p-2 rounded-full hover:bg-gray-100 transition-colors"
              >
                <X className="h-5 w-5 text-gray-600" />
              </button>
            </div>
          </div>

          <div className="bg-blue-50 p-4 rounded-xl mb-6 border border-blue-100">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="bg-blue-100 p-1.5 rounded-lg">
                  {chartOptions.find(opt => opt.type === (autoMode ? recommendedChart : chartType))?.icon || 
                   <BarChart2 className="h-5 w-5 text-blue-600" />}
                </div>
                <div>
                  <h3 className="font-medium text-blue-800">
                    {autoMode ? 
                      `Recommended: ${chartOptions.find(opt => opt.type === recommendedChart)?.label || 'Bar Chart'}` : 
                      chartOptions.find(opt => opt.type === chartType)?.label || 'Bar Chart'}
                  </h3>
                  <p className="text-xs text-blue-600">
                    {chartOptions.find(opt => opt.type === (autoMode ? recommendedChart : chartType))?.description || 
                     'Visualizing your data effectively'}
                  </p>
                </div>
              </div>
              <button 
                onClick={() => setShowOptions(!showOptions)}
                className="bg-white text-sm flex items-center gap-1 px-3 py-1.5 rounded-lg border border-blue-200 text-blue-700 hover:bg-blue-50"
              >
                <Settings className="h-3.5 w-3.5" />
                <span>Change Chart Type</span>
                <ChevronDown className={`h-3.5 w-3.5 transition-transform ${showOptions ? 'rotate-180' : ''}`} />
              </button>
            </div>
          </div>

          {/* Parameters Selection - ALWAYS VISIBLE */}
          <div className="bg-gray-50 p-4 rounded-xl mb-6">
            <h3 className="text-sm font-medium text-gray-700 mb-3">Data Axes Selection</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-1">
                <label className="text-sm font-medium text-gray-600">X-Axis / Categories</label>
                <select
                  value={xAxis}
                  onChange={(e) => setXAxis(e.target.value)}
                  className="w-full border border-gray-200 rounded-lg p-2 bg-white"
                >
                  {availableColumns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-500">
                  {categoricalColumns.includes(xAxis) ? 'Categorical data - Good choice!' : 
                   'Numerical data - Usually better for Y-axis'}
                </p>
              </div>
              <div className="space-y-1">
                <label className="text-sm font-medium text-gray-600">Y-Axis / Values</label>
                <select
                  value={yAxis}
                  onChange={(e) => setYAxis(e.target.value)}
                  className="w-full border border-gray-200 rounded-lg p-2 bg-white"
                >
                  {availableColumns.map(col => (
                    <option key={col} value={col}>{col}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-500">
                  {numericalColumns.includes(yAxis) ? 'Numerical data - Good choice!' : 
                   'Categorical data - Usually better for X-axis'}
                </p>
              </div>
              {(chartType === 'composed' || (autoMode && recommendedChart === 'composed')) && (
                <div className="space-y-1">
                  <label className="text-sm font-medium text-gray-600">Secondary Y-Axis</label>
                  <select
                    value={secondaryYAxis}
                    onChange={(e) => setSecondaryYAxis(e.target.value)}
                    className="w-full border border-gray-200 rounded-lg p-2 bg-white"
                  >
                    {availableColumns.map(col => (
                      <option key={col} value={col}>{col}</option>
                    ))}
                  </select>
                  <p className="text-xs text-gray-500">
                    {numericalColumns.includes(secondaryYAxis) ? 'Numerical data - Good choice!' : 
                     'Categorical data - Not recommended for Y-axis'}
                  </p>
                </div>
              )}
            </div>
          </div>
          
          {showOptions && (
            <div className="mb-6 animate-in fade-in slide-in-from-top-4 duration-300">
              {/* Chart Type Selection */}
              <div className="bg-gray-50 p-4 rounded-xl">
                <h3 className="text-sm font-medium text-gray-600 mb-3">Chart Type</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                  {compatibleCharts.map(option => (
                    <button
                      key={option.type}
                      onClick={() => {
                        setChartType(option.type);
                        setAutoMode(false);
                      }}
                      disabled={!option.compatible}
                      className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all relative ${
                        chartType === option.type && !autoMode 
                          ? 'bg-blue-600 text-white shadow-md' 
                          : option.compatible 
                            ? 'bg-white border border-gray-200 text-gray-700 hover:bg-gray-50' 
                            : 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      } ${option.recommended ? 'border-2 border-green-400' : ''}`}
                    >
                      {option.icon}
                      <span>{option.label}</span>
                      {option.recommended && (
                        <div className="absolute -top-1 -right-1 bg-green-500 text-white p-0.5 rounded-full">
                          <Check className="h-3 w-3" />
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}
          
          {/* Chart Visualization */}
          <div className="bg-gray-50 p-4 rounded-xl h-80 mb-2 flex items-center justify-center">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                {renderAutoChart()}
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500 text-center">
                <p>Select valid axes to display chart</p>
              </div>
            )}
          </div>

          <div className="text-center text-xs text-gray-500">
            <p>You can change the axes and data dimensions regardless of the selected chart type</p>
          </div>
        </div>
      </div>
    </div>
  );
} 