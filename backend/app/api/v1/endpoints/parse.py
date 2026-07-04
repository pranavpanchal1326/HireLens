# ruff: noqa: B008, E501
"""POST /parse endpoint implementation (Phase 7.2).

Exposes a live HTTP route for parsing resumes (PDF/TXT uploads) and job descriptions
(via text input or file upload).
"""

from __future__ import annotations

import logging
import os
import tempfile
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

# Enforce a strict 5MB limit to prevent memory exhaustion / denial of service.
MAX_ALLOWED_SIZE_BYTES = 5 * 1024 * 1024


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
    # 1. Enforce size limit via headers first (if client sent Content-Length)
    if file and file.size is not None and file.size > MAX_ALLOWED_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 5MB.",
        )

    # 2. Read file bytes and check actual size
    file_bytes = b""
    if file:
        file_bytes = await file.read()
        if len(file_bytes) > MAX_ALLOWED_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="File too large. Maximum allowed size is 5MB.",
            )
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

    # 3. Sniﬀ PDF magic bytes to prevent spoofing
    if file and file.filename and file.filename.lower().endswith(".pdf"):
        if not file_bytes.startswith(b"%PDF"):
            raise HTTPException(
                status_code=400,
                detail="Invalid file signature. File content does not match a valid PDF format.",
            )

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
            parsed_resume = structure_resume(extraction_result)
        except UnsupportedFileTypeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            # Clean up immediately after extraction
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

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
