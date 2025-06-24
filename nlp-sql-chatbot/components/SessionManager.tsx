import { useState } from 'react';
import { X, Database, Check, AlertCircle, Loader2 } from 'lucide-react';
import { createSession, SessionRequest } from '../lib/api';

interface SessionManagerProps {
  onSessionCreated: (sessionId: string) => void;
  isOpen: boolean;
  onClose: () => void;
}

export default function SessionManager({ onSessionCreated, isOpen, onClose }: SessionManagerProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<SessionRequest>({
    db_name: 'AdventureWorks',
    username: 'postgres',
    password: 'anmol',
    host: 'localhost',
    port: '5432',
    use_memory: true,
    use_cache: true
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === 'checkbox' ? checked : value
    });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);

    try {
      const response = await createSession(formData);
      onSessionCreated(response.session_id);
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create session');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4 animate-in fade-in duration-300">
      <div className="bg-white rounded-2xl p-6 w-full max-w-lg shadow-2xl border border-gray-200 animate-in zoom-in-95 duration-300">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl">
              <Database className="h-5 w-5 text-white" />
            </div>
            <h2 className="text-xl font-bold text-gray-800">Connect to Database</h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-xl transition-colors duration-200"
            disabled={isLoading}
          >
            <X className="h-5 w-5 text-gray-500" />
          </button>
        </div>
        
        {/* Error Message */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-xl animate-in slide-in-from-top-2 duration-300">
            <div className="flex items-center space-x-2">
              <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
              <p className="text-sm text-red-700 font-medium">{error}</p>
            </div>
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-5">
          {/* Database Name */}
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-gray-700">
              Database Name
            </label>
            <input
              type="text"
              name="db_name"
              value={formData.db_name}
              onChange={handleChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 bg-gray-50 hover:bg-white"
              placeholder="e.g., postgres, mydb"
              disabled={isLoading}
            />
          </div>
          
          {/* Username */}
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-gray-700">
              Username
            </label>
            <input
              type="text"
              name="username"
              value={formData.username}
              onChange={handleChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 bg-gray-50 hover:bg-white"
              disabled={isLoading}
            />
          </div>
          
          {/* Password */}
          <div className="space-y-2">
            <label className="block text-sm font-semibold text-gray-700">
              Password
            </label>
            <input
              type="password"
              name="password"
              value={formData.password}
              onChange={handleChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 bg-gray-50 hover:bg-white"
              disabled={isLoading}
            />
          </div>
          
          {/* Host and Port */}
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-gray-700">
                Host
              </label>
              <input
                type="text"
                name="host"
                value={formData.host}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 bg-gray-50 hover:bg-white"
                disabled={isLoading}
              />
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-semibold text-gray-700">
                Port
              </label>
              <input
                type="text"
                name="port"
                value={formData.port}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 bg-gray-50 hover:bg-white"
                disabled={isLoading}
              />
            </div>
          </div>
          
          {/* Options */}
          <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-xl border border-blue-200">
            <h3 className="text-sm font-semibold text-gray-700 mb-3">Performance Options</h3>
            <div className="flex items-center justify-between">
              <label className="flex items-center space-x-3 cursor-pointer">
                <div className="relative">
                  <input
                    type="checkbox"
                    name="use_memory"
                    checked={formData.use_memory}
                    onChange={handleChange}
                    className="sr-only"
                    disabled={isLoading}
                  />
                  <div className={`w-5 h-5 rounded border-2 transition-all duration-200 ${
                    formData.use_memory 
                      ? 'bg-blue-600 border-blue-600' 
                      : 'border-gray-300 hover:border-blue-400'
                  }`}>
                    {formData.use_memory && (
                      <Check className="h-3 w-3 text-white absolute top-0.5 left-0.5" />
                    )}
                  </div>
                </div>
                <span className="text-sm font-medium text-gray-700">Use Memory</span>
              </label>
              
              <label className="flex items-center space-x-3 cursor-pointer">
                <div className="relative">
                  <input
                    type="checkbox"
                    name="use_cache"
                    checked={formData.use_cache}
                    onChange={handleChange}
                    className="sr-only"
                    disabled={isLoading}
                  />
                  <div className={`w-5 h-5 rounded border-2 transition-all duration-200 ${
                    formData.use_cache 
                      ? 'bg-blue-600 border-blue-600' 
                      : 'border-gray-300 hover:border-blue-400'
                  }`}>
                    {formData.use_cache && (
                      <Check className="h-3 w-3 text-white absolute top-0.5 left-0.5" />
                    )}
                  </div>
                </div>
                <span className="text-sm font-medium text-gray-700">Use Cache</span>
              </label>
            </div>
          </div>
          
          {/* Buttons */}
          <div className="flex justify-end gap-3 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="px-6 py-3 bg-gray-100 hover:bg-gray-200 text-gray-700 font-semibold rounded-xl transition-all duration-200 hover:shadow-md"
              disabled={isLoading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl transition-all duration-200 hover:shadow-lg transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none flex items-center space-x-2"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Connecting...</span>
                </>
              ) : (
                <>
                  <Database className="h-4 w-4" />
                  <span>Connect</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}