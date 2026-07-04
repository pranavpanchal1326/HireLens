import React, { useState } from 'react';
import axios from 'axios';
import { Upload, FileText, CheckCircle2, AlertCircle, RefreshCw } from 'lucide-react';

export default function ResumeParser({ onParsed, parsedData }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setError(null);
      handleParse(selected);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const selected = e.dataTransfer.files[0];
    if (selected) {
      setFile(selected);
      setError(null);
      handleParse(selected);
    }
  };

  const handleParse = async (selectedFile) => {
    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('document_type', 'resume');

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
        setError('Failed to parse the file. Please check if the backend service is running.');
      }
      onParsed(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white dark:bg-[#0c0c0f] border border-zinc-200 dark:border-zinc-800 rounded-xl p-5 shadow-sm">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-zinc-500 mb-3">
        1. Resume Ingestion
      </h3>

      {/* Drag & Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all duration-200
          ${dragOver 
            ? 'border-blue-500 bg-blue-50/10 dark:bg-blue-900/5' 
            : 'border-zinc-200 dark:border-zinc-800 hover:border-zinc-300 dark:hover:border-zinc-700'
          }`}
      >
        <input
          id="resume-upload"
          type="file"
          accept=".pdf,.txt"
          onChange={handleFileChange}
          className="hidden"
          disabled={loading}
        />
        <label htmlFor="resume-upload" className="cursor-pointer block">
          {loading ? (
            <div className="flex flex-col items-center justify-center py-2">
              <RefreshCw className="w-8 h-8 text-blue-500 animate-spin mb-2" />
              <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">Processing Document...</p>
              <p className="text-xs text-zinc-500 mt-1">Applying heuristics and spaCy text extractor</p>
            </div>
          ) : file ? (
            <div className="flex flex-col items-center justify-center py-2">
              <FileText className="w-8 h-8 text-blue-500 mb-2" />
              <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200 truncate max-w-xs">{file.name}</p>
              <p className="text-xs text-zinc-500 mt-1">{(file.size / 1024).toFixed(1)} KB • Click to change</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-2">
              <Upload className="w-8 h-8 text-zinc-400 dark:text-zinc-600 mb-2" />
              <p className="text-sm font-medium text-zinc-800 dark:text-zinc-200">Drag & Drop Resume here</p>
              <p className="text-xs text-zinc-500 mt-1">Supported formats: PDF, TXT (Max 5MB)</p>
            </div>
          )}
        </label>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-4 bg-rose-50 dark:bg-rose-950/20 border border-rose-200 dark:border-rose-800/30 text-rose-800 dark:text-rose-400 rounded-lg p-3.5 flex items-start gap-2.5 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Ingestion Blocked</span>
            {error}
          </div>
        </div>
      )}

      {/* Structured Output Preview */}
      {parsedData && (
        <div className="mt-5 border-t border-zinc-100 dark:border-zinc-800/80 pt-4 animate-fade-in">
          <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 text-xs font-semibold mb-3">
            <CheckCircle2 className="w-4 h-4" /> Document Structured Successfully (Confidence: {Math.round(parsedData.parsing_confidence * 100)}%)
          </div>

          <div className="space-y-3 text-xs">
            {/* Contact details */}
            <div className="bg-zinc-50 dark:bg-zinc-900/30 border border-zinc-100 dark:border-zinc-850 rounded-lg p-3">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block mb-1">Candidate Overview</span>
              <p className="font-semibold text-zinc-800 dark:text-zinc-200 text-sm">
                {parsedData.name || 'Extracted Candidate'}
              </p>
              <div className="grid grid-cols-2 gap-2 mt-1 text-zinc-500 dark:text-zinc-400">
                <p className="truncate">📧 {parsedData.email || 'No email found'}</p>
                <p className="truncate">📞 {parsedData.phone || 'No phone found'}</p>
              </div>
            </div>

            {/* Skills */}
            <div className="bg-zinc-50 dark:bg-zinc-900/30 border border-zinc-100 dark:border-zinc-850 rounded-lg p-3">
              <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wider block mb-2">Technical Skills</span>
              {parsedData.skills && parsedData.skills.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {parsedData.skills.map((skill, i) => (
                    <span key={i} className="bg-white dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 border border-zinc-200 dark:border-zinc-700 px-2 py-0.5 rounded text-[10px] font-mono">
                      {skill}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-zinc-500 italic">No skills extracted.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
