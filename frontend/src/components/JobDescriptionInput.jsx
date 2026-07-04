import React, { useState } from 'react';
import axios from 'axios';
import { FileText, CheckCircle2, AlertCircle, RefreshCw, Clipboard } from 'lucide-react';

export default function JobDescriptionInput({ onParsed, parsedData }) {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('text'); // 'text' or 'file'

  const handleTextChange = (e) => {
    setText(e.target.value);
    setError(null);
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setError(null);
      handleParseFile(selected);
    }
  };

  const handleParseText = async () => {
    if (!text.trim()) {
      setError('Please paste or type the job description.');
      return;
    }
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('jd_text', text);
    formData.append('document_type', 'jd');

    try {
      const response = await axios.post('http://localhost:8000/api/v1/parse', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      onParsed(response.data);
    } catch (err) {
      console.error(err);
      if (err.response && err.response.data && err.response.data.message) {
        setError(err.response.data.message);
      } else {
        setError('Failed to parse the job description. Verify backend status.');
      }
      onParsed(null);
    } finally {
      setLoading(false);
    }
  };

  const handleParseFile = async (selectedFile) => {
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('document_type', 'jd');

    try {
      const response = await axios.post('http://localhost:8000/api/v1/parse', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      onParsed(response.data);
    } catch (err) {
      console.error(err);
      if (err.response && err.response.data && err.response.data.message) {
        setError(err.response.data.message);
      } else {
        setError('Failed to parse the file. Verify backend status.');
      }
      onParsed(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm mt-5">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
          2. Job Description Setup
        </h3>
        {/* Toggle Mode */}
        <div className="bg-zinc-100 dark:bg-[#0c0c0f] p-1 rounded-lg flex gap-1 border border-zinc-200 dark:border-zinc-800">
          <button
            onClick={() => setMode('text')}
            className={`px-3 py-1 rounded text-xs font-medium transition-all duration-150 flex items-center gap-1
              ${mode === 'text' 
                ? 'bg-white dark:bg-zinc-800 text-blue-600 dark:text-blue-400 shadow-sm' 
                : 'text-zinc-550 hover:text-zinc-800 dark:hover:text-zinc-200'
              }`}
          >
            <Clipboard className="w-3.5 h-3.5" /> Paste Text
          </button>
          <button
            onClick={() => setMode('file')}
            className={`px-3 py-1 rounded text-xs font-medium transition-all duration-150 flex items-center gap-1
              ${mode === 'file' 
                ? 'bg-white dark:bg-zinc-800 text-blue-600 dark:text-blue-400 shadow-sm' 
                : 'text-zinc-550 hover:text-zinc-800 dark:hover:text-zinc-200'
              }`}
          >
            <FileText className="w-3.5 h-3.5" /> Upload File
          </button>
        </div>
      </div>

      {mode === 'text' ? (
        <div className="space-y-3">
          <textarea
            value={text}
            onChange={handleTextChange}
            placeholder="Paste the job description or target role requirements here..."
            disabled={loading}
            className="w-full h-28 bg-white dark:bg-[#09090b] border border-zinc-200 dark:border-zinc-800 rounded-lg p-3 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 text-zinc-950 dark:text-zinc-100 transition-all resize-none"
          />
          <button
            onClick={handleParseText}
            disabled={loading || !text.trim()}
            className="w-full bg-[#3b82f6] hover:bg-blue-600 disabled:bg-zinc-100 dark:disabled:bg-zinc-800/80 disabled:text-zinc-400 text-white rounded-lg py-2 text-xs font-medium transition-colors shadow-sm flex items-center justify-center gap-1.5 cursor-pointer disabled:cursor-not-allowed"
          >
            {loading && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
            Parse Job Description
          </button>
        </div>
      ) : (
        <div className="border-2 border-dashed rounded-lg p-6 text-center border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700 transition-all duration-200 cursor-pointer">
          <input
            id="jd-upload"
            type="file"
            accept=".pdf,.txt"
            onChange={handleFileChange}
            className="hidden"
            disabled={loading}
          />
          <label htmlFor="jd-upload" className="cursor-pointer block">
            {loading ? (
              <div className="flex flex-col items-center justify-center">
                <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mb-2" />
                <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">Structuring JD...</p>
              </div>
            ) : file ? (
              <div className="flex flex-col items-center justify-center">
                <FileText className="w-8 h-8 text-blue-500 mb-2" />
                <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate max-w-xs">{file.name}</p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center">
                <FileText className="w-8 h-8 text-zinc-400 dark:text-zinc-600 mb-2" />
                <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">Upload JD file (PDF or TXT)</p>
              </div>
            )}
          </label>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-800/30 text-rose-800 dark:text-rose-400 rounded-lg p-3 flex items-start gap-2 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Validation Failed</span>
            {error}
          </div>
        </div>
      )}

      {/* Structured Output Preview */}
      {parsedData && (
        <div className="mt-5 border-t border-zinc-100 dark:border-zinc-800/80 pt-4 animate-fade-in">
          <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 text-xs font-semibold mb-3">
            <CheckCircle2 className="w-4 h-4" /> Requirements Structured (Confidence: {Math.round(parsedData.parsing_confidence * 100)}%)
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {/* Experience and education */}
            <div className="bg-zinc-50 dark:bg-zinc-900/30 border border-zinc-100 dark:border-zinc-850 rounded-lg p-3">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block mb-1">Target Profile</span>
              <p className="text-zinc-800 dark:text-zinc-200">
                Experience: <span className="font-semibold">{parsedData.required_years_experience || 0} years</span>
              </p>
              <p className="text-zinc-800 dark:text-zinc-200 mt-1">
                Degree: <span className="font-semibold">{parsedData.required_education_level || 'Not Specified'}</span>
              </p>
            </div>

            {/* Required Skills */}
            <div className="bg-zinc-50 dark:bg-zinc-900/30 border border-zinc-100 dark:border-zinc-850 rounded-lg p-3">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block mb-1.5">Required Skills</span>
              <div className="flex flex-wrap gap-1">
                {parsedData.required_skills && parsedData.required_skills.slice(0, 6).map((skill, i) => (
                  <span key={i} className="bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border border-zinc-200 dark:border-zinc-700 px-1.5 py-0.5 rounded text-[9px] font-mono">
                    {skill}
                  </span>
                ))}
                {parsedData.required_skills && parsedData.required_skills.length > 6 && (
                  <span className="text-[9px] text-zinc-400 pt-0.5">+{parsedData.required_skills.length - 6} more</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
