import axios from 'axios';

// Single axios instance for the HireLens backend. Base URL is overridable via
// Vite env for deploys; defaults to the local FastAPI dev server.
const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1',
  timeout: 30000,
});

// Disclosed first-party anon id for freemium rate limiting (backend resolve_anon_id):
// a stable token THIS browser holds — not a covert fingerprint. Persisted so the
// 3-scans/month count is honest across reloads.
function getAnonId() {
  try {
    let id = localStorage.getItem('hirelens_anon_id');
    if (!id) {
      id = (crypto.randomUUID?.() || String(Date.now() + Math.random()));
      localStorage.setItem('hirelens_anon_id', id);
    }
    return id;
  } catch {
    return 'ephemeral';
  }
}
client.interceptors.request.use((config) => {
  config.headers['X-Anon-Id'] = getAnonId();
  return config;
});

// Normalize backend/network errors into a warm, blameless message (§10.10/§12).
// The SYSTEM couldn't do it — never blame the user, never surface a stack trace.
export function humanizeError(err) {
  if (err?.response) {
    const { status, data } = err.response;
    if (status === 429 && data?.reason === 'FREEMIUM_LIMIT_REACHED') {
      return {
        kind: 'freemium',
        message: "You've used your 3 free scans this month.",
        remaining: data.remaining ?? 0,
        resetsAt: data.resets_at || null,
      };
    }
    if (status === 401) {
      return { kind: 'auth', message: "Those recruiter credentials didn't match. Please sign in again." };
    }
    if (status === 400) {
      return { kind: 'input', message: data?.detail || data?.message || "We couldn't read this document clearly." };
    }
    return { kind: 'server', message: data?.message || data?.detail || 'Something went wrong on our side. Please try again.' };
  }
  if (err?.code === 'ECONNABORTED') {
    return { kind: 'timeout', message: 'This is taking longer than expected. Please try again.' };
  }
  return { kind: 'network', message: "We couldn't reach the scoring engine. Check that the backend is running." };
}

// POST /parse (multipart) — resume file → structured parse (incl. parsing_confidence).
export async function parseResume(file) {
  const form = new FormData();
  form.append('file', file);
  form.append('document_type', 'resume');
  const { data } = await client.post('/parse', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data; // ParsedResume
}

// POST /score — the fit computation. Accepts raw text on both sides.
export async function scoreFit({ resumeText, jdText, blindMode = false, photoPresent = false }) {
  const { data } = await client.post('/score', {
    raw_resume_text: resumeText,
    raw_jd_text: jdText,
    blind_mode: blindMode,
    resume_photo_present: photoPresent,
  });
  return data; // { score_result, pipeline_maturity, anonymization? }
}

// POST /rank — recruiter batch ranking (HTTP Basic auth, PRD §9). `resumes` is
// [{ candidate_id, raw_resume_text }]. Batches ≤50 return the ranking synchronously.
export async function rankCandidates({ jdText, resumes, auth }) {
  const { data } = await client.post(
    '/rank',
    { raw_jd_text: jdText, resumes },
    { auth: { username: auth.username, password: auth.password } },
  );
  return data; // RankResponse { ranking_result, pipeline_maturity, total_* , failures }
}

// GET /metrics — recruiter accuracy dashboard feed (HTTP Basic auth). Returns
// readiness_state + current_metrics (may be null while ground truth is collected).
export async function getMetrics({ auth }) {
  const { data } = await client.get('/metrics', {
    auth: { username: auth.username, password: auth.password },
  });
  return data;
}

// Lightweight credential probe: a ranking with an empty batch would 422, so we
// validate creds by attempting a tiny real rank is wasteful — instead the first
// real /rank surfaces a 401 which humanizeError maps. Exposed for the sign-in form.
export function isAuthError(err) {
  return err?.response?.status === 401;
}

export default client;
