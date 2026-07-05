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
    <div className="bg-surface border border-border rounded-xl p-5 shadow-sm">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted mb-3">
        1. Resume Ingestion
      </h3>

      {/* Drag & Drop Zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all duration-200
          ${dragOver
            ? 'border-ember-500 bg-ember-50'
            : 'border-border hover:border-ember-300'
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
              <RefreshCw className="w-8 h-8 text-ember-500 animate-spin mb-2" />
              <p className="text-sm font-medium text-ink">Processing Document...</p>
              <p className="text-xs text-muted mt-1">Applying heuristics and spaCy text extractor</p>
            </div>
          ) : file ? (
            <div className="flex flex-col items-center justify-center py-2">
              <FileText className="w-8 h-8 text-ember-500 mb-2" />
              <p className="text-sm font-medium text-ink truncate max-w-xs">{file.name}</p>
              <p className="text-xs text-muted mt-1 tabular-nums">{(file.size / 1024).toFixed(1)} KB • Click to change</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-2">
              <Upload className="w-8 h-8 text-muted mb-2" />
              <p className="text-sm font-medium text-ink">Drag & Drop Resume here</p>
              <p className="text-xs text-muted mt-1">Supported formats: PDF, TXT (Max 5MB)</p>
            </div>
          )}
        </label>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-4 bg-gap-fill border border-gap-500 text-gap-text rounded-lg p-3.5 flex items-start gap-2.5 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Ingestion Blocked</span>
            {error}
          </div>
        </div>
      )}

      {/* Structured Output Preview */}
      {parsedData && (
        <div className="mt-5 border-t border-border pt-4 animate-fade-in">
          <div className="flex items-center gap-1.5 text-fit-text text-xs font-semibold mb-3">
            <CheckCircle2 className="w-4 h-4" /> Document Structured Successfully (Confidence: <span className="tabular-nums">{Math.round(parsedData.parsing_confidence * 100)}%</span>)
          </div>

          <div className="space-y-3 text-xs">
            {/* Contact details */}
            <div className="bg-canvas border border-border rounded-lg p-3">
              <span className="text-[10px] font-semibold text-muted uppercase tracking-wider block mb-1">Candidate Overview</span>
              <p className="font-semibold text-ink text-sm">
                {parsedData.name || 'Extracted Candidate'}
              </p>
              <div className="grid grid-cols-2 gap-2 mt-1 text-muted">
                <p className="truncate">📧 {parsedData.email || 'No email found'}</p>
                <p className="truncate">📞 {parsedData.phone || 'No phone found'}</p>
              </div>
            </div>

            {/* Skills */}
            <div className="bg-canvas border border-border rounded-lg p-3">
              <span className="text-[10px] font-semibold text-muted uppercase tracking-wider block mb-2">Technical Skills</span>
              {parsedData.skills && parsedData.skills.length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {parsedData.skills.map((skill, i) => (
                    <span key={i} className="bg-surface text-ink border border-border px-2 py-0.5 rounded text-[10px]">
                      {skill}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-muted italic">No skills extracted.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
