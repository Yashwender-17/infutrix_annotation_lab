"""
token_usage_logger_local.py
────────────────────────────
Standalone, self-contained token usage logger for the Annotation Lab.
Logs ONLY to a local CSV file — no GCS, no DB, no external dependencies.

Designed to be a drop-in for annotators who don't have the full
Gradesmith infrastructure.
"""

import csv
import os
import inspect
import contextvars
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# LOG FILE — written next to this script in logs/
# ─────────────────────────────────────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)
TOKEN_LOG_FILE = _LOG_DIR / "annotation_lab_token_usage.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT VARIABLES
# ─────────────────────────────────────────────────────────────────────────────
_ctx_annotator   = contextvars.ContextVar('annotator',   default='')
_ctx_image_name  = contextvars.ContextVar('image_name',  default='')
_ctx_session_id  = contextvars.ContextVar('session_id',  default='')


def set_ocr_context(annotator: str = '', image_name: str = '', session_id: str = ''):
    """
    Call before each OCR request so log entries carry meaningful context.

    Args:
        annotator:  Name of the person doing annotation (e.g. 'Raunak')
        image_name: Filename of the image being OCR-ed
        session_id: Optional batch/session identifier
    """
    _ctx_annotator.set(str(annotator))
    _ctx_image_name.set(str(image_name))
    _ctx_session_id.set(str(session_id))


def clear_ocr_context():
    """Reset context variables."""
    _ctx_annotator.set('')
    _ctx_image_name.set('')
    _ctx_session_id.set('')


# ─────────────────────────────────────────────────────────────────────────────
# PRICING TABLE
# ─────────────────────────────────────────────────────────────────────────────
def _get_pricing(model_name: str):
    """Returns (input_$/M, output_$/M) for the given model."""
    m = model_name.lower()
    if "gemini-2.5-flash-lite" in m:
        return (0.10, 0.40)
    elif "gemini-2.5-flash" in m:
        return (0.30, 2.50)
    elif "gemini-2.5-pro" in m:
        return (1.25, 10.00)
    elif "gemini-3.1-flash-lite" in m:
        return (0.25, 1.50)
    # Fallback
    return (0.30, 2.50)


# ─────────────────────────────────────────────────────────────────────────────
# CSV HEADERS
# ─────────────────────────────────────────────────────────────────────────────
_CSV_HEADERS = [
    'Timestamp', 'Annotator', 'Image Name', 'Session ID',
    'Calling File', 'Calling Function', 'Model',
    'Prompt Tokens', 'Output Tokens', 'Thoughts Tokens', 'Total Tokens',
    'Input Cost ($)', 'Output Cost ($)', 'Total Cost ($)',
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOGGING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def log_token_usage(response_obj, model_name: str):
    """
    Logs Gemini token usage to a local CSV file.

    Args:
        response_obj: The Gemini API response object (has .usage_metadata)
        model_name:   Model identifier string (e.g. 'gemini-2.5-flash-lite')
    """
    try:
        usage = getattr(response_obj, 'usage_metadata', None)

        prompt_tokens   = getattr(usage, 'prompt_token_count',     0) or 0
        candidate_tokens= getattr(usage, 'candidates_token_count', 0) or 0
        total_tokens    = getattr(usage, 'total_token_count',       0) or 0
        thoughts_tokens = getattr(usage, 'thoughts_token_count',    0) or 0

        # Output = total - input (thoughts are part of output budget)
        output_tokens = max(total_tokens - prompt_tokens, 0)

        # Cost
        input_ppm, output_ppm = _get_pricing(model_name)
        input_cost  = (prompt_tokens  / 1_000_000) * input_ppm
        output_cost = (output_tokens  / 1_000_000) * output_ppm
        total_cost  = input_cost + output_cost

        # Caller detection — walk up stack, skip this file and genai_client.py
        stack = inspect.stack()
        caller_file = 'unknown'
        caller_func = 'unknown'
        skip_files  = {'token_usage_logger_local.py', 'genai_client.py'}
        for frame in stack[1:]:
            fname = Path(frame.filename).name
            if fname not in skip_files:
                caller_file = fname
                caller_func = frame.function
                break

        # Context
        annotator  = _ctx_annotator.get()  or 'N/A'
        image_name = _ctx_image_name.get() or 'N/A'
        session_id = _ctx_session_id.get() or 'N/A'

        # Write to CSV
        file_exists = TOKEN_LOG_FILE.exists()
        with open(TOKEN_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(_CSV_HEADERS)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                annotator,
                image_name,
                session_id,
                caller_file,
                caller_func,
                model_name,
                prompt_tokens,
                output_tokens,
                thoughts_tokens,
                total_tokens,
                f"{input_cost:.6f}",
                f"{output_cost:.6f}",
                f"{total_cost:.6f}",
            ])

    except Exception as e:
        # Never crash the calling code — just warn
        print(f"⚠️  [token_logger] Failed to log token usage: {e}")
