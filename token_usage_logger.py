import csv
import os
import inspect
import contextvars
from datetime import datetime
from pathlib import Path
from config.yaml_loader import PATHS
from utils.db_log_writer import insert_token_usage
from utils.gcs_utils import upload_file_to_gcs, download_file_from_gcs

TOKEN_LOG_FILE = Path(PATHS.LOGS) / "Gemini_token_usage_log.csv"

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT VARIABLES — set by the grading worker, read by log_token_usage()
# These flow automatically through async calls without touching function signatures.
# ─────────────────────────────────────────────────────────────────────────────
_ctx_upload_id = contextvars.ContextVar('upload_id', default='')
_ctx_exam_id = contextvars.ContextVar('exam_id', default='')
_ctx_roll_number = contextvars.ContextVar('roll_number', default='')
_ctx_course_code = contextvars.ContextVar('course_code', default='')
_ctx_cumulative_cost = contextvars.ContextVar('cumulative_cost', default=0.0)


def set_grading_context(upload_id='', exam_id='', roll_number='', course_code=''):
    """
    Call this once at the start of processing a student upload.
    All subsequent log_token_usage() calls will automatically include this context.
    """
    _ctx_upload_id.set(str(upload_id))
    _ctx_exam_id.set(str(exam_id))
    _ctx_roll_number.set(str(roll_number))
    _ctx_course_code.set(str(course_code))
    _ctx_cumulative_cost.set(0.0)


def clear_grading_context():
    """Reset context after processing is done."""
    _ctx_upload_id.set('')
    _ctx_exam_id.set('')
    _ctx_roll_number.set('')
    _ctx_course_code.set('')
    _ctx_cumulative_cost.set(0.0)


def get_cumulative_cost() -> float:
    """Returns the cumulative API cost for the current upload context."""
    return _ctx_cumulative_cost.get()


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STAGE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _detect_pipeline_stage(caller_file: str, caller_function: str) -> str:
    """Auto-detect which pipeline stage made the API call based on caller info."""
    file_lower = caller_file.lower()
    func_lower = caller_function.lower()

    if 'question_answer_mapping_pipeline' in file_lower or 'mapping' in file_lower:
        if 'process_page' in func_lower or 'ocr' in func_lower:
            return 'OCR'
        elif 'map_answers' in func_lower or 'mapping' in func_lower:
            return 'Mapping'
        else:
            return 'Mapping Pipeline'
    elif 'grader' in file_lower or 'grading' in file_lower:
        return 'Grading'
    elif 'calculator' in file_lower:
        return 'Score Calculator'
    elif 'extraction' in file_lower or 'extract' in file_lower:
        return 'Question Extraction'
    elif 'instruction' in file_lower:
        return 'Instruction Generation'
    elif 'sheet_seg' in file_lower or 'segmentation' in file_lower:
        return 'Sheet Segmentation'
    elif 'marks_hierarchy' in file_lower or 'marks_validation' in file_lower:
        return 'Marks Validation'
    elif 'genai_client' in file_lower:
        return 'Direct API Call'
    else:
        return 'Other'


# ─────────────────────────────────────────────────────────────────────────────
# PRICING
# ─────────────────────────────────────────────────────────────────────────────
def get_pricing_for_model(model_name: str):
    """
    Returns (input_cost_per_million, output_cost_per_million) based on model.
    """
    m = model_name.lower()

    if "gemini-2.5-flash-lite" in m:
        return (0.1, 0.4)  # input, output
    elif "gemini-2.5-flash" in m:
        return (0.30, 2.5)
    elif "gemini-2.5-pro" in m:
        return (1.25, 10.00)
    elif "gemini-3.1-flash-lite-preview" in m:
        return (0.25, 1.5)  # global endpoint
    elif "gemini-3.1-flash-lite" in m:
        return (0.25, 1.5)   #global

    # Default: flash pricing
    return (0.30, 2.5)


# ─────────────────────────────────────────────────────────────────────────────
# CSV HEADERS
# ─────────────────────────────────────────────────────────────────────────────
CSV_HEADERS = [
    'Timestamp', 'Upload ID', 'Exam ID', 'Roll Number', 'Course Code',
    'Pipeline Stage', 'Calling File', 'Calling Function', 'Model',
    'Prompt Tokens', 'Candidate Tokens', 'Output Tokens',
    'Thoughts Tokens', 'Cache Tokens', 'Total Tokens',
    'Input Cost ($)', 'Output Cost ($)', 'Total Cost ($)'
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOGGING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def log_token_usage(response_obj, model_name: str):
    """
    Logs token usage with full grading context (upload_id, roll_number, etc.)
    and auto-detected pipeline stage.
    """

    try:
        usage = getattr(response_obj, 'usage_metadata', None)

        prompt_tokens = getattr(usage, 'prompt_token_count', 0) or 0
        candidate_tokens = getattr(usage, 'candidates_token_count', 0) or 0
        total_tokens = getattr(usage, 'total_token_count', 0) or 0
        
        thoughts_tokens = getattr(usage, 'thoughts_token_count', None)
        cache_tokens = getattr(usage, 'cache_tokens_details', None)
        
        # Output tokens = total - input
        output_tokens = max(total_tokens - prompt_tokens, 0)

        # ── Cost calculation ──
        input_price_per_million, output_price_per_million = get_pricing_for_model(model_name)
        actual_input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
        actual_output_cost = (output_tokens / 1_000_000) * output_price_per_million
        actual_total_cost = actual_input_cost + actual_output_cost


        # ── Detect caller ──
        # We walk up the stack to find the first caller that is NOT in genai_client.py or token_usage_logger.py
        stack = inspect.stack()
        caller_frame = None
        for frame in stack[1:]:
            filename = Path(frame.filename).name
            if filename not in ['genai_client.py', 'token_usage_logger.py']:
                caller_frame = frame
                break
        
        if not caller_frame:
            # Fallback if we can't find an external caller
            caller_frame = stack[min(len(stack)-1, 2)]
            
        caller_function = caller_frame.function
        caller_file = Path(caller_frame.filename).name

        # ── Auto-detect pipeline stage ──
        pipeline_stage = _detect_pipeline_stage(caller_file, caller_function)

        # ── Read grading context ──
        # We use .get() with a default fallback to ensure no empty strings are written to CSV
        raw_upload_id = _ctx_upload_id.get()
        raw_exam_id = _ctx_exam_id.get()
        raw_roll_number = _ctx_roll_number.get()
        raw_course_code = _ctx_course_code.get()

        upload_id = raw_upload_id if raw_upload_id and str(raw_upload_id).strip() else "N/A"
        exam_id = raw_exam_id if raw_exam_id and str(raw_exam_id).strip() else "N/A"
        roll_number = raw_roll_number if raw_roll_number and str(raw_roll_number).strip() else "N/A"
        course_code = raw_course_code if raw_course_code and str(raw_course_code).strip() else "N/A"

        # ── Write to CSV ──
        file_exists = os.path.isfile(TOKEN_LOG_FILE)

        db_payload = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "upload_id": upload_id,
            "exam_id": exam_id,
            "roll_number": roll_number,
            "course_code": course_code,
            "pipeline_stage": pipeline_stage,
            "calling_file": caller_file,
            "calling_function": caller_function,
            "model": model_name,
            "prompt_tokens": prompt_tokens if prompt_tokens is not None else 0,
            "candidate_tokens": candidate_tokens if candidate_tokens is not None else 0,
            "output_tokens": output_tokens if output_tokens is not None else 0,
            "thoughts_tokens": thoughts_tokens if thoughts_tokens is not None else 0,
            "cache_tokens": cache_tokens if cache_tokens is not None else "N/A",
            "total_tokens": total_tokens if total_tokens is not None else 0,
            "input_cost": f"{actual_input_cost:.6f}",
            "output_cost": f"{actual_output_cost:.6f}",
            "total_cost": f"{actual_total_cost:.6f}",
        }

        if insert_token_usage(db_payload):
            return

        gcs_rel_path = "Gemini_token_usage_log.csv"
        # Download from GCS first to keep it in sync
        download_file_from_gcs(gcs_rel_path, str(TOKEN_LOG_FILE))

        with open(TOKEN_LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(CSV_HEADERS)

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                upload_id,
                exam_id,
                roll_number,
                course_code,

                pipeline_stage,
                caller_file,
                caller_function,
                model_name,

                prompt_tokens if prompt_tokens is not None else 0,
                candidate_tokens if candidate_tokens is not None else 0,
                output_tokens if output_tokens is not None else 0,
                thoughts_tokens if thoughts_tokens is not None else 0,
                cache_tokens if cache_tokens is not None else "N/A",
                total_tokens if total_tokens is not None else 0,

                f"{actual_input_cost:.6f}",
                f"{actual_output_cost:.6f}",
                f"{actual_total_cost:.6f}"
            ])
            
            # Upload back to GCS
            upload_file_to_gcs(str(TOKEN_LOG_FILE), gcs_rel_path)

    except Exception as e:
        print(f"⚠️ Failed to log tokens: {e}")


# --------------------------------------------------
