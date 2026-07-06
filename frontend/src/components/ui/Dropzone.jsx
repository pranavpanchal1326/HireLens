import { useRef, useState } from 'react';
import { UploadCloud, FileText, CheckCircle2 } from 'lucide-react';
import ArcMeter from './ArcMeter';

// §10.2 Resume upload dropzone. Large, rounded (r-xl), dashed border, warm empty
// state. Drag-over: ember-50 fill + ember-300 dashed border. Critically, once
// parsed it shows the PARSING-CONFIDENCE result — how many expected fields were
// extracted — so a bad input is visible before it ever reaches scoring (§10.9).
export default function Dropzone({
  onFile,
  accept = '.pdf,.docx,.txt',
  parsing = false,
  parsed = null,        // { fileName, parseConfidence: 0-1, fieldsFound, fieldsExpected }
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const pick = (file) => file && onFile?.(file);

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    pick(e.dataTransfer.files?.[0]);
  };

  // Parsed → confidence-forward summary card.
  if (parsed) {
    const conf = parsed.parseConfidence ?? 0;
    const tone = conf >= 0.8 ? 'fit' : conf >= 0.5 ? 'gap' : 'lowconf';
    return (
      <div className="rounded-xl border border-border bg-surface p-5">
        <div className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3 min-w-0">
            <span className="w-10 h-10 rounded-lg bg-fit-fill text-fit-text flex items-center justify-center shrink-0">
              <CheckCircle2 className="w-5 h-5" strokeWidth={1.75} />
            </span>
            <div className="min-w-0">
              <p className="text-small font-medium text-ink truncate">{parsed.fileName}</p>
              <p className="text-caption text-muted">
                Read {parsed.fieldsFound}/{parsed.fieldsExpected} expected fields
              </p>
            </div>
          </div>
          <ArcMeter value={conf} tone={tone} size={52} strokeWidth={6} />
        </div>
        <button
          onClick={() => inputRef.current?.click()}
          className="text-caption text-ember-700 hover:underline mt-3 focus-ember"
        >
          Replace file
        </button>
        <input ref={inputRef} type="file" accept={accept} className="hidden"
          onChange={(e) => pick(e.target.files?.[0])} />
      </div>
    );
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`rounded-[var(--r-xl)] border-2 border-dashed p-8 text-center cursor-pointer transition-colors duration-200 focus-ember ${
        dragging ? 'border-ember-300 bg-ember-50' : 'border-border bg-surface hover:bg-canvas'
      }`}
    >
      <span className="w-12 h-12 mx-auto rounded-2xl bg-ember-50 text-ember-700 flex items-center justify-center mb-3">
        {parsing ? (
          <FileText className="w-6 h-6 animate-pulse" strokeWidth={1.75} />
        ) : (
          <UploadCloud className="w-6 h-6" strokeWidth={1.75} />
        )}
      </span>
      {parsing ? (
        <>
          <p className="text-small font-medium text-ink">Reading your resume…</p>
          <p className="text-caption text-muted mt-1">Extracting fields and checking readability</p>
        </>
      ) : (
        <>
          <p className="text-small font-medium text-ink">Drop your resume, or click to browse</p>
          <p className="text-caption text-muted mt-1">PDF, DOCX, or TXT · read locally, not stored beyond your session</p>
        </>
      )}
      <input ref={inputRef} type="file" accept={accept} className="hidden"
        onChange={(e) => pick(e.target.files?.[0])} />
    </div>
  );
}
