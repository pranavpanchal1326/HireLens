import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, Trash2, ListOrdered, Users } from 'lucide-react';
import Button from '../../components/ui/Button';
import Card from '../../components/ui/Card';
import { Input, Textarea } from '../../components/ui/Field';
import { ErrorState } from '../../components/ui/states';
import { useRecruiter } from '../../recruiter/RecruiterContext';
import { rankCandidates, humanizeError } from '../../lib/api';

const SAMPLE_JD =
  'Senior Backend Engineer. Requirements: 5+ years Python, REST API design, PostgreSQL, Kubernetes, Terraform, and team leadership. AWS preferred. Bachelor degree in Computer Science required.';

const SAMPLE_POOL = [
  { name: 'Jane Doe', text: 'Senior Backend Engineer, 6 years Python microservices, REST APIs, PostgreSQL, Docker. Led 4 engineers, cut deploy time 40%. Kubernetes, Terraform, AWS. B.S. Computer Science.' },
  { name: 'Alex Rivera', text: 'Backend developer, 3 years Python and Flask, some PostgreSQL. Built internal REST tools. Bootcamp graduate. Eager to learn Kubernetes.' },
  { name: 'Priya Nair', text: 'Staff Engineer, 9 years. Python, Go, PostgreSQL, Kubernetes, Terraform, AWS at scale. Led platform team of 8. M.S. Computer Science.' },
  { name: 'Sam Cole', text: 'Data analyst transitioning to backend. SQL, some Python scripting, Excel. No Kubernetes or cloud infra yet. B.A. Economics.' },
];

let cid = 0;
const emptyRow = () => ({ id: ++cid, name: '', text: '' });

export default function Batch() {
  const { auth, setRanking, signOut } = useRecruiter();
  const navigate = useNavigate();
  const [jdText, setJdText] = useState('');
  const [rows, setRows] = useState([emptyRow(), emptyRow(), emptyRow()]);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);

  const filled = rows.filter((r) => r.text.trim().length > 40);
  const canRank = jdText.trim().length > 40 && filled.length >= 2 && status !== 'ranking';

  const update = (id, patch) => setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  const addRow = () => setRows((rs) => [...rs, emptyRow()]);
  const removeRow = (id) => setRows((rs) => (rs.length > 1 ? rs.filter((r) => r.id !== id) : rs));

  const loadSample = () => {
    setJdText(SAMPLE_JD);
    setRows(SAMPLE_POOL.map((c) => ({ id: ++cid, name: c.name, text: c.text })));
  };

  async function handleRank() {
    setStatus('ranking');
    setError(null);
    const resumes = filled.map((r, i) => ({
      candidate_id: r.name.trim() || `Candidate ${i + 1}`,
      raw_resume_text: r.text,
    }));
    try {
      const data = await rankCandidates({ jdText, resumes, auth });
      setRanking(data);
      navigate('/recruiter/ranked');
    } catch (err) {
      const e = humanizeError(err);
      setError(e);
      setStatus('error');
      if (e.kind === 'auth') signOut();
    }
  }

  return (
    <div className="max-w-[1000px] mx-auto px-6 py-10">
      <div className="flex items-end justify-between gap-4 mb-6">
        <div>
          <h1 className="text-h1 text-ink">Rank a candidate pool</h1>
          <p className="text-body text-muted mt-1">One job description, many resumes — ranked and explained in seconds.</p>
        </div>
        <Button variant="ghost" size="sm" icon={Users} onClick={loadSample}>Load sample pool</Button>
      </div>

      <Card pad="lg" className="mb-5">
        <Textarea id="batch-jd" label="Job description" rows={4} value={jdText}
          onChange={(e) => setJdText(e.target.value)} placeholder="Paste the role's job description…" />
      </Card>

      <div className="flex items-center justify-between mb-3">
        <h2 className="text-h3 text-ink">Candidates <span className="text-muted tabular-nums">({filled.length})</span></h2>
        <Button variant="secondary" size="sm" icon={Plus} onClick={addRow}>Add candidate</Button>
      </div>

      <div className="space-y-3">
        {rows.map((r, i) => (
          <Card key={r.id} pad="sm">
            <div className="flex items-start gap-3">
              <span className="w-7 h-7 rounded-md bg-canvas border border-border text-muted text-caption font-semibold tabular-nums flex items-center justify-center shrink-0 mt-1">
                {i + 1}
              </span>
              <div className="flex-1 space-y-2">
                <Input id={`cand-${r.id}`} value={r.name} onChange={(e) => update(r.id, { name: e.target.value })}
                  placeholder={`Candidate ${i + 1} name (optional)`} className="h-9" />
                <Textarea id={`cand-text-${r.id}`} rows={3} value={r.text}
                  onChange={(e) => update(r.id, { text: e.target.value })} placeholder="Paste resume text…" />
              </div>
              <button onClick={() => removeRow(r.id)} className="text-muted hover:text-gap-text p-1.5 mt-1 focus-ember rounded" aria-label="Remove candidate">
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          </Card>
        ))}
      </div>

      {error && (
        <div className="mt-5"><ErrorState title="We couldn't run this ranking" body={error.message} /></div>
      )}

      <div className="mt-6 flex items-center justify-between">
        <p className="text-caption text-muted">Add at least 2 candidates. Batches up to 50 rank instantly.</p>
        <Button variant="primary" size="lg" icon={ListOrdered} disabled={!canRank} loading={status === 'ranking'} onClick={handleRank}>
          {status === 'ranking' ? 'Ranking…' : `Rank ${filled.length} candidates`}
        </Button>
      </div>
    </div>
  );
}
