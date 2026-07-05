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
    <div className="bg-surface border border-border rounded-xl p-5 shadow-sm mt-5">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">
          2. Job Description Setup
        </h3>
        {/* Toggle Mode */}
        <div className="bg-canvas p-1 rounded-lg flex gap-1 border border-border">
          <button
            onClick={() => setMode('text')}
            className={`px-3 py-1 rounded text-xs font-medium transition-all duration-150 flex items-center gap-1
              ${mode === 'text'
                ? 'bg-surface text-ember-700 shadow-sm'
                : 'text-muted hover:text-ink'
              }`}
          >
            <Clipboard className="w-3.5 h-3.5" /> Paste Text
          </button>
          <button
            onClick={() => setMode('file')}
            className={`px-3 py-1 rounded text-xs font-medium transition-all duration-150 flex items-center gap-1
              ${mode === 'file'
                ? 'bg-surface text-ember-700 shadow-sm'
                : 'text-muted hover:text-ink'
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
            className="w-full h-28 bg-surface border border-border rounded-lg p-3 text-xs shadow-sm focus:outline-none focus:ring-2 focus:ring-ember-500 focus:border-ember-500 text-ink transition-all resize-none"
          />
          <button
            onClick={handleParseText}
            disabled={loading || !text.trim()}
            className="w-full bg-ember-500 hover:bg-ember-700 disabled:bg-canvas disabled:text-muted text-white rounded-lg py-2 text-xs font-medium transition-colors shadow-sm flex items-center justify-center gap-1.5 cursor-pointer disabled:cursor-not-allowed"
          >
            {loading && <RefreshCw className="w-3.5 h-3.5 animate-spin" />}
            Parse Job Description
          </button>
        </div>
      ) : (
        <div className="border-2 border-dashed rounded-lg p-6 text-center border-border hover:border-ember-300 transition-all duration-200 cursor-pointer">
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
                <RefreshCw className="w-8 h-8 text-ember-500 animate-spin mb-2" />
                <p className="text-sm font-medium text-ink">Structuring JD...</p>
              </div>
            ) : file ? (
              <div className="flex flex-col items-center justify-center">
                <FileText className="w-8 h-8 text-ember-500 mb-2" />
                <p className="text-sm font-medium text-ink truncate max-w-xs">{file.name}</p>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center">
                <FileText className="w-8 h-8 text-muted mb-2" />
                <p className="text-sm font-medium text-ink">Upload JD file (PDF or TXT)</p>
              </div>
            )}
          </label>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mt-4 bg-gap-fill border border-gap-500 text-gap-text rounded-lg p-3 flex items-start gap-2 text-xs">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold block">Validation Failed</span>
            {error}
          </div>
        </div>
      )}

      {/* Structured Output Preview */}
      {parsedData && (
        <div className="mt-5 border-t border-border pt-4 animate-fade-in">
          <div className="flex items-center gap-1.5 text-fit-text text-xs font-semibold mb-3">
            <CheckCircle2 className="w-4 h-4" /> Requirements Structured (Confidence: <span className="tabular-nums">{Math.round(parsedData.parsing_confidence * 100)}%</span>)
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            {/* Experience and education */}
            <div className="bg-canvas border border-border rounded-lg p-3">
              <span className="text-[10px] font-semibold text-muted uppercase tracking-wider block mb-1">Target Profile</span>
              <p className="text-ink">
                Experience: <span className="font-semibold tabular-nums">{parsedData.required_years_experience || 0} years</span>
              </p>
              <p className="text-ink mt-1">
                Degree: <span className="font-semibold">{parsedData.required_education_level || 'Not Specified'}</span>
              </p>
            </div>

            {/* Required Skills */}
            <div className="bg-canvas border border-border rounded-lg p-3">
              <span className="text-[10px] font-semibold text-muted uppercase tracking-wider block mb-1.5">Required Skills</span>
              <div className="flex flex-wrap gap-1">
                {parsedData.required_skills && parsedData.required_skills.slice(0, 6).map((skill, i) => (
                  <span key={i} className="bg-surface text-ink border border-border px-1.5 py-0.5 rounded text-[9px]">
                    {skill}
                  </span>
                ))}
                {parsedData.required_skills && parsedData.required_skills.length > 6 && (
                  <span className="text-[9px] text-muted pt-0.5 tabular-nums">+{parsedData.required_skills.length - 6} more</span>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
