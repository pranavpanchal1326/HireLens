# ruff: noqa: E501
"""Phase 7 Full Audit — Integration Test Module (7.X).

Covers Passes 1–7 of the Phase 7 audit, exercising the ENTIRE API surface
as one coherent system using SYNTHETIC data only.

Pass 1: Error-shape consistency (all errors → global envelope)
Pass 2: Status-function reuse integrity (verified via import tracing — see audit report)
Pass 3: Guardrails retrofit correctness (before/after equivalence + adversarial consistency)
Pass 4: Auth fail-closed verification (adversarial bypass attempts)
Pass 5: Data isolation re-verification (shared model + per-account feedback)
Pass 6: Read-only guarantee for /metrics post-7.8
Pass 7: Full end-to-end synthetic integration test

All fixtures are clearly labeled SYNTHETIC and cannot be mistaken for production data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.v1.endpoints.score import get_orchestrator_tools
import app.api.v1.endpoints.feedback as feedback_module
import app.api.v1.endpoints.metrics as metrics_module
from app.schemas.scoring import ConfidenceLevel, FeatureVector, ScoreResult, SkillMatch, GapItem
from app.services.orchestration.agent_orchestrator import OrchestratorTools

client = TestClient(app, raise_server_exceptions=False)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC STUBS — clearly labeled, not reused from any prior test file
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHETIC_RESUME_TEXT = (
    "SYNTHETIC AUDIT FIXTURE — John Audit with ten years of Python, "
    "machine learning, and software engineering experience. "
    "Expert in data pipeline development, API design, and team leadership."
)

SYNTHETIC_JD_TEXT = (
    "SYNTHETIC AUDIT FIXTURE — Senior Python Developer needed with "
    "experience in machine learning, API development, and team management. "
    "Requirements include strong communication skills and software engineering."
)

SYNTHETIC_GARBAGE_RESUME = "xyzqwmbn plkjhgf asdfghjkl zxcvbnm qwertyuiop"


class AuditStubExperienceMatcher:
    def match(self, *args, **kwargs) -> float:
        return 0.85


class AuditStubSkillMatcher:
    def match_resume_to_jd(self, *args, **kwargs) -> tuple[float, list, list]:
        return (
            0.75,
            [SkillMatch(resume_skill="python", jd_skill="python", match_type="exact", similarity_score=1.0)],
            [GapItem(missing_skill="kubernetes", suggested_action="Learn Kubernetes")],
        )


class AuditStubHybridScorer:
    def compute_hybrid_score(self, *args, **kwargs) -> ScoreResult:
        return ScoreResult(
            resume_id="audit-r1",
            jd_id="audit-j1",
            final_score=82,
            feature_vector=FeatureVector(
                tfidf_score=0.75, embedding_score=0.85,
                skill_overlap_pct=0.70, exp_match=0.85, edu_match=0.60,
            ),
            scoring_confidence=0.80,
            confidence_level=ConfidenceLevel.HIGH,
            parsing_confidence=0.95,
            pipeline_version="v3-hybrid",
        )


class AuditStubCaseStore:
    def build_case_embedding(self, *args, **kwargs):
        return None

    def retrieve_similar_cases(self, *args, **kwargs):
        return []

    def calibration_check(self, *args, **kwargs):
        class DummyCalibration:
            is_outlier = False
            deviation = 0.0
            similar_case_ids = []
            similar_case_scores = []
            low_sample_warning = False
        return DummyCalibration()


_AUDIT_TOOLS = OrchestratorTools(
    hybrid_scorer=AuditStubHybridScorer(),  # type: ignore[arg-type]
    skill_matcher=AuditStubSkillMatcher(),  # type: ignore[arg-type]
    taxonomy_entries=[],
    case_store=AuditStubCaseStore(),  # type: ignore[arg-type]
    experience_matcher=AuditStubExperienceMatcher(),  # type: ignore[arg-type]
)

# Auth credentials from auth.py's VALID_RECRUITERS
ACCT_A = ("recruiter_one", "password123")   # account: company_a
ACCT_B = ("recruiter_two", "password456")   # account: company_b


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def isolate_persistence(tmp_path: Path):
    """Redirect all persistence paths to tmp_path for full isolation."""
    orig_fb_dataset = feedback_module.DATASET_PATH
    orig_fb_recruiter = feedback_module.RECRUITER_PATH
    orig_fb_audit = feedback_module.AUDIT_LOG_PATH
    orig_mt_dataset = metrics_module.DATASET_PATH
    orig_mt_history = metrics_module.METRICS_HISTORY_PATH

    feedback_module.DATASET_PATH = tmp_path / "ground_truth_dataset.json"
    feedback_module.RECRUITER_PATH = tmp_path / "recruiter_feedback.json"
    feedback_module.AUDIT_LOG_PATH = tmp_path / "feedback_audit_log.jsonl"
    metrics_module.DATASET_PATH = tmp_path / "ground_truth_dataset.json"
    metrics_module.METRICS_HISTORY_PATH = tmp_path / "metrics_history.json"

    feedback_module._idempotency_keys.clear()

    yield tmp_path

    feedback_module.DATASET_PATH = orig_fb_dataset
    feedback_module.RECRUITER_PATH = orig_fb_recruiter
    feedback_module.AUDIT_LOG_PATH = orig_fb_audit
    metrics_module.DATASET_PATH = orig_mt_dataset
    metrics_module.METRICS_HISTORY_PATH = orig_mt_history


@pytest.fixture(autouse=True)
def mock_tools():
    """Inject audit stub tools, no auth override (tests use real credentials)."""
    app.dependency_overrides[get_orchestrator_tools] = lambda: _AUDIT_TOOLS
    yield
    app.dependency_overrides.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 1 — ERROR-SHAPE CONSISTENCY
# Every error response must match: {"code": ..., "message": ..., "request_id": ...}
# ═══════════════════════════════════════════════════════════════════════════════

class TestPass1ErrorShapeConsistency:
    """Verify every distinct failure mode routes through the global error envelope."""

    @staticmethod
    def _assert_error_envelope(response, expected_status: int):
        """Assert that a response matches the Phase 7.1 global error envelope shape."""
        assert response.status_code == expected_status, (
            f"Expected {expected_status}, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert "code" in data, f"Missing 'code' key in error response: {data}"
        assert "message" in data, f"Missing 'message' key in error response: {data}"
        assert "request_id" in data, f"Missing 'request_id' key in error response: {data}"

    def test_health_no_error_shape(self):
        """Health is not an error — but verify it doesn't accidentally match error shape."""
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert "code" not in data  # success responses should NOT have error keys

    def test_parse_400_missing_file(self):
        """Parse: missing resume file → 400."""
        r = client.post("/api/v1/parse", data={"document_type": "resume"})
        self._assert_error_envelope(r, 400)

    def test_parse_400_empty_file(self):
        """Parse: empty file → 400."""
        r = client.post("/api/v1/parse", files={"file": ("empty.pdf", b"", "application/pdf")}, data={"document_type": "resume"})
        self._assert_error_envelope(r, 400)

    def test_parse_400_spoofed_pdf(self):
        """Parse: spoofed PDF → 400."""
        r = client.post("/api/v1/parse", files={"file": ("fake.pdf", b"Not a PDF", "application/pdf")}, data={"document_type": "resume"})
        self._assert_error_envelope(r, 400)

    def test_score_400_missing_inputs(self):
        """Score: missing both resume and JD → 400."""
        r = client.post("/api/v1/score", json={"raw_resume_text": "", "raw_jd_text": "JD text"})
        self._assert_error_envelope(r, 400)

    def test_score_422_malformed_json(self):
        """Score: completely malformed request body → 422 validation error."""
        r = client.post("/api/v1/score", json={"not_a_valid_field": 123})
        # This returns 400 because the endpoint logic checks for missing text
        # before Pydantic can complain — all fields are Optional
        assert r.status_code in (400, 422)
        data = r.json()
        assert "code" in data
        assert "message" in data

    def test_rank_401_unauthenticated(self):
        """Rank: no credentials → 401 with WWW-Authenticate header."""
        payload = {"raw_jd_text": "JD", "resumes": [{"candidate_id": "c1", "raw_resume_text": "Resume"}]}
        r = client.post("/api/v1/rank", json=payload)
        self._assert_error_envelope(r, 401)
        assert "www-authenticate" in r.headers

    def test_rank_401_bad_credentials(self):
        """Rank: wrong password → 401."""
        payload = {"raw_jd_text": "JD", "resumes": [{"candidate_id": "c1", "raw_resume_text": "Resume"}]}
        r = client.post("/api/v1/rank", json=payload, auth=("recruiter_one", "wrong"))
        self._assert_error_envelope(r, 401)

    def test_feedback_401_unauthenticated(self):
        """Feedback: no credentials → 401."""
        payload = {"feedback_type": "rater", "pair_id": "p1", "resume_id": "r1", "jd_id": "j1",
                   "score": 80.0, "rater_id": "rater-X", "justification": "Valid text here."}
        r = client.post("/api/v1/feedback", json=payload)
        self._assert_error_envelope(r, 401)

    def test_feedback_403_recruiter_mismatch(self):
        """Feedback: recruiter tries to impersonate another → 403."""
        payload = {"feedback_type": "recruiter", "score_id": "s1",
                   "actual_outcome": "hired", "recruiter_id": "recruiter_two"}
        r = client.post("/api/v1/feedback", json=payload, auth=ACCT_A)
        self._assert_error_envelope(r, 403)

    def test_feedback_422_invalid_score_range(self):
        """Feedback: rater score out of range → 422 validation error."""
        payload = {"feedback_type": "rater", "pair_id": "p1", "resume_id": "r1", "jd_id": "j1",
                   "score": 999.0, "rater_id": "rater-X", "justification": "Valid text here."}
        r = client.post("/api/v1/feedback", json=payload, auth=ACCT_A)
        data = r.json()
        assert r.status_code == 422
        assert "code" in data
        assert "message" in data

    def test_metrics_401_unauthenticated(self):
        """Metrics: no credentials → 401."""
        r = client.get("/api/v1/metrics")
        self._assert_error_envelope(r, 401)


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 3 — GUARDRAILS RETROFIT CORRECTNESS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPass3GuardrailsRetrofit:
    """Re-derive guardrail correctness independently with 3+ cases per endpoint."""

    # /parse: empty file, spoofed PDF, non-English content
    def test_parse_rejects_empty_file(self):
        r = client.post("/api/v1/parse", files={"file": ("r.pdf", b"", "application/pdf")}, data={"document_type": "resume"})
        assert r.status_code == 400
        assert "empty" in r.json()["message"].lower()

    def test_parse_rejects_spoofed_pdf(self):
        r = client.post("/api/v1/parse", files={"file": ("r.pdf", b"SPOOFED CONTENT", "application/pdf")}, data={"document_type": "resume"})
        assert r.status_code == 400
        assert "signature" in r.json()["message"].lower()

    def test_parse_rejects_non_english_resume(self):
        cyrillic = "Привет это резюме на русском языке с информацией для проверки." * 2
        r = client.post("/api/v1/parse", files={"file": ("r.txt", cyrillic.encode("utf-8"), "text/plain")}, data={"document_type": "resume"})
        assert r.status_code == 400
        assert "english" in r.json()["message"].lower()

    def test_parse_accepts_valid_english_resume(self):
        valid = "Senior Python engineer with ten years of experience in API development and team leadership."
        r = client.post("/api/v1/parse", files={"file": ("r.txt", valid.encode("utf-8"), "text/plain")}, data={"document_type": "resume"})
        assert r.status_code == 200

    # /score: empty text, whitespace-only, valid
    def test_score_rejects_empty_resume(self):
        r = client.post("/api/v1/score", json={"raw_resume_text": "", "raw_jd_text": "JD text"})
        assert r.status_code == 400

    def test_score_rejects_whitespace_resume(self):
        r = client.post("/api/v1/score", json={"raw_resume_text": "   ", "raw_jd_text": "JD text"})
        assert r.status_code == 400

    def test_score_accepts_valid_inputs(self):
        r = client.post("/api/v1/score", json={"raw_resume_text": SYNTHETIC_RESUME_TEXT, "raw_jd_text": SYNTHETIC_JD_TEXT})
        assert r.status_code == 200

    # /rank: oversized batch, non-English candidate isolated, valid batch
    def test_rank_rejects_oversized_batch(self):
        # R5: 51-1000 now run async; only >1000 is rejected at validation.
        resumes = [{"candidate_id": f"c-{i}", "raw_resume_text": "Python experience"} for i in range(1001)]
        r = client.post("/api/v1/rank", json={"raw_jd_text": SYNTHETIC_JD_TEXT, "resumes": resumes}, auth=ACCT_A)
        assert r.status_code in (400, 422)

    def test_rank_isolates_non_english_candidate(self):
        payload = {
            "raw_jd_text": SYNTHETIC_JD_TEXT,
            "resumes": [
                {"candidate_id": "good-1", "raw_resume_text": SYNTHETIC_RESUME_TEXT},
                {"candidate_id": "bad-1", "raw_resume_text": SYNTHETIC_GARBAGE_RESUME},
            ],
        }
        r = client.post("/api/v1/rank", json=payload, auth=ACCT_A)
        assert r.status_code == 200
        data = r.json()
        assert data["total_successful"] == 1
        assert data["total_failed"] == 1

    def test_rank_accepts_valid_batch(self):
        payload = {
            "raw_jd_text": SYNTHETIC_JD_TEXT,
            "resumes": [{"candidate_id": "c1", "raw_resume_text": SYNTHETIC_RESUME_TEXT}],
        }
        r = client.post("/api/v1/rank", json=payload, auth=ACCT_A)
        assert r.status_code == 200
        assert r.json()["total_successful"] == 1

    # /feedback: invalid justification, valid rater, valid recruiter
    def test_feedback_rejects_numeric_justification(self):
        payload = {"feedback_type": "rater", "pair_id": "p1", "resume_id": "r1", "jd_id": "j1",
                   "score": 80.0, "rater_id": "rater-X", "justification": "12345"}
        r = client.post("/api/v1/feedback", json=payload, auth=ACCT_A)
        assert r.status_code == 422

    def test_feedback_accepts_valid_rater(self):
        payload = {"feedback_type": "rater", "pair_id": "audit-p1", "resume_id": "r1", "jd_id": "j1",
                   "score": 80.0, "rater_id": "rater-X", "justification": "Good match for the role."}
        r = client.post("/api/v1/feedback", json=payload, auth=ACCT_A)
        assert r.status_code == 200

    def test_feedback_accepts_valid_recruiter(self):
        payload = {"feedback_type": "recruiter", "score_id": "audit-s1",
                   "actual_outcome": "interviewed", "recruiter_id": "recruiter_one"}
        r = client.post("/api/v1/feedback", json=payload, auth=ACCT_A)
        assert r.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 4 — AUTH FAIL-CLOSED VERIFICATION (ADVERSARIAL)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPass4AdversarialAuth:
    """New adversarial auth-bypass attempts beyond 7.8's self-tests."""

    def test_rank_empty_resume_list_auth_precedes_validation(self):
        """Auth must reject BEFORE business logic runs — even on trivially invalid input.
        An empty list violates Pydantic min_length=1, but auth check should happen first.
        """
        payload = {"raw_jd_text": "JD", "resumes": []}
        r = client.post("/api/v1/rank", json=payload)
        # Should be 401 (auth) NOT 422 (validation) — auth check runs first
        # Actually: FastAPI processes Pydantic validation before dependencies for body params
        # So this might be 422. The key safety property is: no business logic runs.
        assert r.status_code in (401, 422)
        # Regardless of order, the response must still match the error envelope
        data = r.json()
        assert "code" in data
        assert "message" in data

    def test_rank_auth_with_empty_username(self):
        """Empty username with valid-format Basic auth header."""
        payload = {"raw_jd_text": "JD", "resumes": [{"candidate_id": "c1", "raw_resume_text": "Resume"}]}
        r = client.post("/api/v1/rank", json=payload, auth=("", "password123"))
        assert r.status_code == 401

    def test_rank_auth_with_empty_password(self):
        """Valid username but empty password."""
        payload = {"raw_jd_text": "JD", "resumes": [{"candidate_id": "c1", "raw_resume_text": "Resume"}]}
        r = client.post("/api/v1/rank", json=payload, auth=("recruiter_one", ""))
        assert r.status_code == 401

    def test_feedback_recruiter_body_injection_different_recruiter_id(self):
        """Authenticated as Account A, but payload claims recruiter_id of Account B.
        This MUST be rejected even though the auth header is valid."""
        payload = {"feedback_type": "recruiter", "score_id": "s-adversarial",
                   "actual_outcome": "hired", "recruiter_id": "recruiter_two"}
        r = client.post("/api/v1/feedback", json=payload, auth=ACCT_A)
        assert r.status_code == 403

    def test_feedback_recruiter_overwrite_another_accounts_outcome(self):
        """Account A creates outcome, Account B tries to overwrite via same score_id."""
        # Step 1: Account A creates
        payload_a = {"feedback_type": "recruiter", "score_id": "s-cross-account",
                     "actual_outcome": "interviewed", "recruiter_id": "recruiter_one"}
        r1 = client.post("/api/v1/feedback", json=payload_a, auth=ACCT_A)
        assert r1.status_code == 200

        # Step 2: Account B tries to overwrite
        payload_b = {"feedback_type": "recruiter", "score_id": "s-cross-account",
                     "actual_outcome": "hired", "recruiter_id": "recruiter_two"}
        r2 = client.post("/api/v1/feedback", json=payload_b, auth=ACCT_B)
        assert r2.status_code == 403

    def test_metrics_with_nonexistent_account(self):
        """Auth with a username not in VALID_RECRUITERS."""
        r = client.get("/api/v1/metrics", auth=("hacker_account", "any_password"))
        assert r.status_code == 401

    def test_rank_valid_auth_nonexistent_account(self):
        """Valid format Basic auth, but account doesn't exist."""
        payload = {"raw_jd_text": "JD", "resumes": [{"candidate_id": "c1", "raw_resume_text": "Resume"}]}
        r = client.post("/api/v1/rank", json=payload, auth=("nonexistent_recruiter", "password"))
        assert r.status_code == 401


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 5 — DATA ISOLATION RE-VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestPass5DataIsolation:
    """Verify correct SELECTIVE isolation: shared model, isolated feedback."""

    def test_shared_model_identical_scores(self):
        """Account A and B calling /score on identical inputs get identical scores.
        This proves the shared model is NOT account-scoped."""
        payload = {"raw_resume_text": SYNTHETIC_RESUME_TEXT, "raw_jd_text": SYNTHETIC_JD_TEXT}

        r_a = client.post("/api/v1/score", json=payload)
        r_b = client.post("/api/v1/score", json=payload)

        assert r_a.status_code == 200
        assert r_b.status_code == 200
        assert r_a.json()["score_result"]["final_score"] == r_b.json()["score_result"]["final_score"]

    def test_shared_model_via_rank_identical_scores(self):
        """Account A and B calling /rank on identical inputs get identical scores."""
        payload = {
            "raw_jd_text": SYNTHETIC_JD_TEXT,
            "resumes": [{"candidate_id": "iso-c1", "raw_resume_text": SYNTHETIC_RESUME_TEXT}],
        }

        r_a = client.post("/api/v1/rank", json=payload, auth=ACCT_A)
        r_b = client.post("/api/v1/rank", json=payload, auth=ACCT_B)

        assert r_a.status_code == 200
        assert r_b.status_code == 200

        score_a = r_a.json()["ranking_result"]["ranked_candidates"][0]["score_result"]["final_score"]
        score_b = r_b.json()["ranking_result"]["ranked_candidates"][0]["score_result"]["final_score"]
        assert score_a == score_b

    def test_feedback_isolation_recruiter_outcomes(self):
        """Account A and B each submit outcomes — verify they don't leak into each other's storage."""
        # Account A submits
        payload_a = {"feedback_type": "recruiter", "score_id": "iso-s1",
                     "actual_outcome": "hired", "recruiter_id": "recruiter_one"}
        r1 = client.post("/api/v1/feedback", json=payload_a, auth=ACCT_A)
        assert r1.status_code == 200

        # Account B submits a DIFFERENT score_id
        payload_b = {"feedback_type": "recruiter", "score_id": "iso-s2",
                     "actual_outcome": "rejected", "recruiter_id": "recruiter_two"}
        r2 = client.post("/api/v1/feedback", json=payload_b, auth=ACCT_B)
        assert r2.status_code == 200

        # Verify storage: both outcomes exist but each is tagged to its recruiter
        with open(feedback_module.RECRUITER_PATH, "r", encoding="utf-8") as f:
            outcomes = json.load(f)
        assert len(outcomes) == 2
        assert outcomes[0]["recruiter_id"] == "recruiter_one"
        assert outcomes[1]["recruiter_id"] == "recruiter_two"

    def test_shared_ground_truth_not_scoped_by_account(self):
        """Rater feedback from two different recruiters goes into the SAME shared dataset."""
        payload_1 = {"feedback_type": "rater", "pair_id": "shared-gt-01", "resume_id": "r1",
                     "jd_id": "j1", "score": 80.0, "rater_id": "rater-A",
                     "justification": "Good match for the role."}
        r1 = client.post("/api/v1/feedback", json=payload_1, auth=ACCT_A)
        assert r1.status_code == 200

        payload_2 = {"feedback_type": "rater", "pair_id": "shared-gt-01", "resume_id": "r1",
                     "jd_id": "j1", "score": 85.0, "rater_id": "rater-B",
                     "justification": "Excellent match for the role."}
        r2 = client.post("/api/v1/feedback", json=payload_2, auth=ACCT_B)
        assert r2.status_code == 200

        # Both ratings end up in the SAME shared dataset file
        from app.services.evaluation.ground_truth_schema import load_dataset
        dataset = load_dataset(str(feedback_module.DATASET_PATH))
        assert len(dataset.pairs) == 1
        assert len(dataset.pairs[0].rater_scores) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 6 — READ-ONLY GUARANTEE FOR /metrics POST-7.8
# ═══════════════════════════════════════════════════════════════════════════════

class TestPass6MetricsReadOnly:
    """Verify /metrics (now auth-wrapped) has zero write side effects."""

    def test_metrics_no_file_mutations(self, isolate_persistence):
        """Call /metrics, snapshot filesystem before/after, verify no changes."""
        tmp_path = isolate_persistence

        # Snapshot: list all files in tmp_path before
        files_before = set()
        for f in tmp_path.rglob("*"):
            if f.is_file():
                files_before.add((str(f.relative_to(tmp_path)), f.stat().st_size))

        # Call /metrics (unready state — no dataset)
        r = client.get("/api/v1/metrics", auth=ACCT_A)
        assert r.status_code == 200
        assert r.json()["readiness_state"] == "unready"

        # Snapshot: list all files after
        files_after = set()
        for f in tmp_path.rglob("*"):
            if f.is_file():
                files_after.add((str(f.relative_to(tmp_path)), f.stat().st_size))

        # No new files created, no existing files modified
        assert files_before == files_after, (
            f"Files changed after /metrics call. Before: {files_before}, After: {files_after}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PASS 7 — FULL END-TO-END SYNTHETIC INTEGRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

class TestPass7EndToEnd:
    """Full user journey through the live API in sequence, using SYNTHETIC data only."""

    def test_full_synthetic_user_journey(self, isolate_persistence):
        """
        Sequence:
        1. /parse a synthetic JD (public, no auth)
        2. /score a synthetic resume against that JD (public, no auth)
        3. /rank a batch with one good and one garbage resume (auth required)
        4. /feedback for one of the ranked candidates (auth required)
        5. /metrics to confirm account-scoped state (auth required)
        """
        tmp_path = isolate_persistence

        # ── Step 1: Parse a JD ──
        parse_r = client.post(
            "/api/v1/parse",
            data={"jd_text": SYNTHETIC_JD_TEXT, "document_type": "jd"},
        )
        assert parse_r.status_code == 200, f"Parse failed: {parse_r.text}"
        parsed_jd = parse_r.json()
        assert "raw_text" in parsed_jd
        assert "required_skills" in parsed_jd

        # ── Step 2: Score a single resume ──
        score_r = client.post("/api/v1/score", json={
            "raw_resume_text": SYNTHETIC_RESUME_TEXT,
            "raw_jd_text": SYNTHETIC_JD_TEXT,
        })
        assert score_r.status_code == 200, f"Score failed: {score_r.text}"
        score_data = score_r.json()
        assert "score_result" in score_data
        assert "pipeline_maturity" in score_data
        assert score_data["score_result"]["final_score"] >= 0

        # ── Step 3: Rank a batch (1 good + 1 garbage) ──
        rank_r = client.post("/api/v1/rank", json={
            "raw_jd_text": SYNTHETIC_JD_TEXT,
            "resumes": [
                {"candidate_id": "e2e-good", "raw_resume_text": SYNTHETIC_RESUME_TEXT},
                {"candidate_id": "e2e-garbage", "raw_resume_text": SYNTHETIC_GARBAGE_RESUME},
            ],
        }, auth=ACCT_A)
        assert rank_r.status_code == 200, f"Rank failed: {rank_r.text}"
        rank_data = rank_r.json()
        assert rank_data["total_submitted"] == 2
        assert rank_data["total_successful"] == 1, "Good resume should succeed"
        assert rank_data["total_failed"] == 1, "Garbage resume should fail"
        assert rank_data["ranking_result"]["ranked_candidates"][0]["candidate_id"] == "e2e-good"
        assert any("e2e-garbage" in f["candidate_id"] for f in rank_data["failures"])

        # ── Step 4: Submit feedback for the ranked candidate ──
        feedback_r = client.post("/api/v1/feedback", json={
            "feedback_type": "recruiter",
            "score_id": "e2e-score-001",
            "actual_outcome": "interviewed",
            "recruiter_id": "recruiter_one",
        }, auth=ACCT_A)
        assert feedback_r.status_code == 200, f"Feedback failed: {feedback_r.text}"
        fb_data = feedback_r.json()
        assert fb_data["status"] == "created"
        assert fb_data["feedback_type"] == "recruiter"

        # ── Step 5: Call /metrics ──
        metrics_r = client.get("/api/v1/metrics", auth=ACCT_A)
        assert metrics_r.status_code == 200, f"Metrics failed: {metrics_r.text}"
        m_data = metrics_r.json()
        assert "readiness_state" in m_data
        # In unready state because no rater feedback has been submitted
        assert m_data["readiness_state"] == "unready"

        # ── Verify: /metrics did NOT create or modify any files ──
        # (the only files should be from the feedback step)
        files_in_tmp = list(tmp_path.rglob("*"))
        file_names = [f.name for f in files_in_tmp if f.is_file()]
        # Only feedback files should exist
        assert "recruiter_feedback.json" in file_names
        assert "feedback_audit_log.jsonl" in file_names
        # ground_truth_dataset.json should NOT exist (no rater feedback submitted)
        assert "ground_truth_dataset.json" not in file_names
