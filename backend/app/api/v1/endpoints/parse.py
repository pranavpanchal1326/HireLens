# ruff: noqa: B008, E501
"""POST /parse endpoint implementation (Phase 7.2).

Exposes a live HTTP route for parsing resumes (PDF/TXT uploads) and job descriptions
(via text input or file upload).
"""

from __future__ import annotations

import logging
import os
import tempfile

from app.api.v1.guardrails import detect_content_quality, validate_file_upload
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.parsing import ExtractionResult, ParsedJobDescription, ParsedResume
from app.services.extraction.pdf_extractor import (
    PDFTextExtractor,
    UnsupportedFileTypeError,
)
from app.services.structuring.nlp_pipeline import (
    structure_job_description,
    structure_resume,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/parse", response_model=None)
async def parse_document(
    file: UploadFile | None = File(None),
    jd_text: str | None = Form(None),
    document_type: Literal["resume", "jd"] = Form("resume"),
) -> ParsedResume | ParsedJobDescription:
    """Parses a resume (from file upload) or job description (from text or file upload).

    PII Lifecycle and Transient Storage:
        - Uploaded file bytes are held in memory temporarily.
        - If processed, they are written to a transient temp file on disk inside a try/finally block.
        - The temp file is unlinked (deleted) immediately after extraction completes.
        - No raw text or PII is written to persistent logs, databases, or cache.
    """
    # 1. Read file bytes and validate via shared guardrails module (Phase 7.7)
    file_bytes = b""
    if file:
        file_bytes = await file.read()
        vr = validate_file_upload(file_bytes, file.filename)
        if not vr.is_valid:
            raise HTTPException(status_code=vr.http_status, detail=vr.error_detail)

    # 4. Resolve routing based on document_type
    if document_type == "resume":
        if not file:
            raise HTTPException(
                status_code=400,
                detail="File upload is required for resume parsing.",
            )

        # Write to transient temp file for processing
        suffix = (
            ".pdf"
            if file.filename and file.filename.lower().endswith(".pdf")
            else ".txt"
        )
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            extractor = PDFTextExtractor()
            extraction_result = extractor.extract(tmp_path)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            # Clean up immediately after extraction
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        # Content quality gate (Phase 7.7 — closes PRD §8.2 non-English gap)
        cq = detect_content_quality(extraction_result.raw_text)
        if not cq.is_acceptable:
            raise HTTPException(status_code=400, detail=cq.reason)

        parsed_resume = structure_resume(extraction_result)
        return parsed_resume

    elif document_type == "jd":
        # JD text can come from form field or file extract
        raw_text = ""
        warnings_list = []

        if file:
            suffix = (
                ".pdf"
                if file.filename and file.filename.lower().endswith(".pdf")
                else ".txt"
            )
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                extractor = PDFTextExtractor()
                extraction_result = extractor.extract(tmp_path)
                raw_text = extraction_result.raw_text
                warnings_list = list(extraction_result.warnings)
            except UnsupportedFileTypeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            finally:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        elif jd_text:
            raw_text = jd_text
        else:
            raise HTTPException(
                status_code=400,
                detail="Either file or jd_text must be provided for job description parsing.",
            )

        if not raw_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Job description text is empty.",
            )

        extraction_result = ExtractionResult(
            raw_text=raw_text,
            extraction_method_used="plain_text",
            warnings=warnings_list,
            is_processable=True,
            page_count=1,
        )

        parsed_jd = structure_job_description(extraction_result)
        return parsed_jd

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported document_type: '{document_type}'",
        )
