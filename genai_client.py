import atexit
import logging
import asyncio
import random
import threading
import weakref
from datetime import datetime
from typing import Optional, List, Union, Any, Dict, Tuple
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import time
import inspect

from google import genai
from google.genai import types
from PIL import Image as PILImage

from .token_usage_logger import log_token_usage
from utils.gcs_utils import upload_file_to_gcs


# ---------------------------------------------------------
# CONSTANTS & CONFIGURATION
# ---------------------------------------------------------
DEFAULT_PROJECT = "gradesmith-demo"
DEFAULT_MODEL = "gemini-2.5-flash"
ACTIVE_GCP_PROJECT = DEFAULT_PROJECT
GCP_PROJECT_POOL = ["gradesmith-demo"]


def set_active_gcp_project(project: str):
    global ACTIVE_GCP_PROJECT
    ACTIVE_GCP_PROJECT = project
    logging.info(f"✅ Global GCP Project set to: {ACTIVE_GCP_PROJECT}")
MAX_RETRIES = 6
RETRY_DELAY = 2.0          # seconds — base for exponential backoff
MAX_RETRY_DELAY = 30.0     # seconds — cap so attempt 6 doesn't sleep ~64s
RETRY_JITTER_RANGE = (0.5, 1.5)  # multiplied into each delay to break thundering herd
REQUEST_TIMEOUT = 300  # 5 minutes


# ---------------------------------------------------------
# REGION SELECTOR BASED ON MODEL NAME
# ---------------------------------------------------------
def get_location_for_model(model: str) -> str:
    """
    Auto-select correct Vertex AI region based on model name.
    
    Args:
        model: The model identifier
        
    Returns:
        Region string for Vertex AI
    """
    m = model.lower()

    if "gemini-2.5-flash-lite" in m:
        return "europe-west8"
    elif "gemini-2.5-flash" in m:
        return "asia-south1"
    elif "gemini-3.1-flash-lite-preview" in m:
        return "global"
    elif "gemini-3.1-flash-lite" in m:
        return "global"

    # Default fallback
    return "asia-south1"


# ---------------------------------------------------------
# ---------------------------------------------------------
# RETRY DECORATOR WITH EXPONENTIAL BACKOFF
# ---------------------------------------------------------
def retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_DELAY):
    """
    Decorator for retrying functions with exponential backoff.
    If the failure is due to MAX_TOKENS, it retries with gemini-2.5-flash as the fallback model.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_args = list(args)
            current_kwargs = dict(kwargs)
            
            for attempt in range(max_retries):
                try:
                    return func(*current_args, **current_kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check for MAX_TOKENS, empty response, SAFETY, RECITATION, or 503 error
                    is_fallback_error = (
                        isinstance(e, ValueError) and (
                            "MAX_TOKENS" in str(e) or 
                            "No candidates" in str(e) or
                            "SAFETY" in str(e) or
                            "RECITATION" in str(e)
                        )
                    ) or (
                        "503" in str(e) or 
                        "UNAVAILABLE" in str(e)
                    )
                    
                    if is_fallback_error:
                        try:
                            sig = inspect.signature(func)
                            bound = sig.bind(*current_args, **current_kwargs)
                            bound.apply_defaults()
                            current_model = bound.arguments.get('model')
                            
                            # If model is not already gemini-2.5-flash, fallback to it
                            if current_model != "gemini-2.5-flash":
                                page_info = ""
                                page_num = bound.arguments.get('page_number')
                                if page_num is not None:
                                    page_info = f" on page {page_num}"
                                
                                context_label = bound.arguments.get('context_label')
                                ctx_str = f" [{context_label}]" if context_label else ""
                                
                                error_type = "MAX_TOKENS" if "MAX_TOKENS" in str(e) else ("SAFETY" if "SAFETY" in str(e) else ("RECITATION" if "RECITATION" in str(e) else ("503 UNAVAILABLE" if "503" in str(e) or "UNAVAILABLE" in str(e) else "Empty Response")))
                                logging.warning(
                                    f"⚠️ {error_type} error detected{page_info}{ctx_str} for model '{current_model}' in {func.__name__}. "
                                    f"Falling back to 'gemini-2.5-flash' for subsequent retry attempts."
                                )
                                bound.arguments['model'] = "gemini-2.5-flash"
                                # Convert back to args and kwargs
                                current_args = list(bound.args)
                                current_kwargs = dict(bound.kwargs)
                        except Exception as sig_err:
                            logging.error(f"⚠️ Failed to update model argument in retry: {sig_err}")
                    
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), MAX_RETRY_DELAY)
                        delay *= random.uniform(*RETRY_JITTER_RANGE)
                        context_label = current_kwargs.get('context_label', '')
                        ctx_str = f" [{context_label}]" if context_label else ""
                        logging.warning(
                            f"⚠️ Attempt {attempt + 1}/{max_retries} failed{ctx_str} [Project: {ACTIVE_GCP_PROJECT}]: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logging.error(f"🔴 All {max_retries} attempts failed [Project: {ACTIVE_GCP_PROJECT}]")

            raise last_exception
        return wrapper
    return decorator


def async_retry_with_backoff(max_retries: int = MAX_RETRIES, base_delay: float = RETRY_DELAY):
    """
    Async decorator for retrying functions with exponential backoff.
    If the failure is due to MAX_TOKENS, it retries with gemini-2.5-flash as the fallback model.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_args = list(args)
            current_kwargs = dict(kwargs)
            
            for attempt in range(max_retries):
                try:
                    return await func(*current_args, **current_kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check for MAX_TOKENS, empty response, SAFETY, RECITATION, or 503 error
                    is_fallback_error = (
                        isinstance(e, ValueError) and (
                            "MAX_TOKENS" in str(e) or 
                            "No candidates" in str(e) or
                            "SAFETY" in str(e) or
                            "RECITATION" in str(e)
                        )
                    ) or (
                        "503" in str(e) or 
                        "UNAVAILABLE" in str(e)
                    )
                    
                    if is_fallback_error:
                        try:
                            sig = inspect.signature(func)
                            bound = sig.bind(*current_args, **current_kwargs)
                            bound.apply_defaults()
                            current_model = bound.arguments.get('model')
                            
                            # If model is not already gemini-2.5-flash, fallback to it
                            if current_model != "gemini-2.5-flash":
                                page_info = ""
                                page_num = bound.arguments.get('page_number')
                                if page_num is not None:
                                    page_info = f" on page {page_num}"
                                    
                                context_label = bound.arguments.get('context_label')
                                ctx_str = f" [{context_label}]" if context_label else ""
                                
                                error_type = "MAX_TOKENS" if "MAX_TOKENS" in str(e) else ("SAFETY" if "SAFETY" in str(e) else ("RECITATION" if "RECITATION" in str(e) else ("503 UNAVAILABLE" if "503" in str(e) or "UNAVAILABLE" in str(e) else "Empty Response")))
                                logging.warning(
                                    f"⚠️ {error_type} error detected{page_info}{ctx_str} for model '{current_model}' in {func.__name__}. "
                                    f"Falling back to 'gemini-2.5-flash' for subsequent retry attempts."
                                )
                                bound.arguments['model'] = "gemini-2.5-flash"
                                # Convert back to args and kwargs
                                current_args = list(bound.args)
                                current_kwargs = dict(bound.kwargs)
                        except Exception as sig_err:
                            logging.error(f"⚠️ Failed to update model argument in retry: {sig_err}")
                    
                    if attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), MAX_RETRY_DELAY)
                        delay *= random.uniform(*RETRY_JITTER_RANGE)
                        context_label = current_kwargs.get('context_label', '')
                        ctx_str = f" [{context_label}]" if context_label else ""
                        logging.warning(
                            f"⚠️ Attempt {attempt + 1}/{max_retries} failed{ctx_str} [Project: {ACTIVE_GCP_PROJECT}]: {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logging.error(f"🔴 All {max_retries} attempts failed [Project: {ACTIVE_GCP_PROJECT}]")
            
            raise last_exception
        return wrapper
    return decorator


# ---------------------------------------------------------
# CLIENT SINGLETON CACHE (per worker process)
# ---------------------------------------------------------
# Vertex genai.Client is thread-safe and supports both sync (client.models)
# and async (client.aio.models) usage. The SDK explicitly recommends
# constructing it ONCE per process and reusing it; constructing a fresh
# Client on every call performs a token-exchange handshake that counts
# against the DSQ pool and is the dominant cause of 429s under bursty load.
_client_cache: Dict[Tuple[str, str], "genai.Client"] = {}
_client_cache_lock = threading.Lock()

# Async clients are cached PER EVENT LOOP. The Vertex async transport binds an
# internal asyncio.Lock to the loop it is first used on; reusing one cached client
# across multiple loops (e.g. repeated asyncio.run() calls in a long-lived API
# process) raises "<Lock> is bound to a different event loop". Keying by the
# running loop gives each loop its own client. Long-lived per-process flows
# (e.g. the grading worker subprocess) have exactly one loop, so this is a no-op
# for them. The sync cache above is intentionally left unchanged.
_async_client_cache: Dict[Tuple[str, str, int], Tuple["asyncio.AbstractEventLoop", "genai.Client"]] = {}
_async_client_cache_lock = threading.Lock()

# Loops on which we've installed the benign-teardown exception filter (below),
# tracked weakly so closed/garbage-collected loops drop out automatically.
_loops_with_exc_filter: "weakref.WeakSet" = weakref.WeakSet()


def _install_async_loop_exception_filter(loop: "asyncio.AbstractEventLoop") -> None:
    """Silence the benign 'Event loop is closed' noise from genai/httpx teardown.

    The Vertex async client opens httpx connections bound to the loop it first
    runs on. With repeated asyncio.run() (one loop per request/item), a previous
    request's client is finalized while a *new* loop is running: its aclose()
    coroutine gets scheduled on the new loop but ultimately calls .close() on the
    original, now-closed loop's transport, raising
    RuntimeError('Event loop is closed'). Nobody awaits that orphan task, so
    asyncio logs a noisy "Task exception was never retrieved" traceback — even
    though the actual Vertex AI request already succeeded.

    We install a per-loop exception handler that swallows exactly this case and
    delegates everything else to the previous/default handler. Installed once per
    loop, from the async call path, so every loop that does Gemini work is
    covered before any such orphan task can fire.
    """
    if loop in _loops_with_exc_filter:
        return

    previous_handler = loop.get_exception_handler()

    def _handler(lp: "asyncio.AbstractEventLoop", context: dict) -> None:
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return  # benign cross-loop teardown race — drop it
        if previous_handler is not None:
            previous_handler(lp, context)
        else:
            lp.default_exception_handler(context)

    loop.set_exception_handler(_handler)
    _loops_with_exc_filter.add(loop)


def _get_or_create_client(model: str, project: str) -> "genai.Client":
    """Return a cached genai.Client for (location, project), creating it on first use."""
    location = get_location_for_model(model)
    key = (location, project)

    client = _client_cache.get(key)
    if client is not None:
        return client

    with _client_cache_lock:
        # Double-checked locking: another caller may have created it while we waited.
        client = _client_cache.get(key)
        if client is not None:
            return client

        logging.info(f"🌍 Initializing singleton genai.Client - Location: {location}, Project: {project}")
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(
                api_version='v1',
                # headers={
                #     "X-Vertex-AI-LLM-Request-Type": "shared",  # only relevant with Provisioned Throughput; on-demand is the default
                #     "X-Vertex-AI-LLM-Shared-Request-Type": "priority"  # priority tier costs ~75-100% more per token
                # }
            )
        )
        _client_cache[key] = client
        return client


def _evict_dead_async_clients_locked() -> None:
    """Drop cached async clients whose event loop has been closed.

    Caller must hold _async_client_cache_lock. This prevents clients from
    accumulating as short-lived asyncio.run() loops come and go.

    We intentionally do NOT call genai's ``client.close()`` here: for an async
    client that method schedules an ``aclose()`` coroutine as a Task on the loop
    the client was bound to. Since that loop is already closed, the scheduling
    raises ``RuntimeError: Event loop is closed`` and asyncio logs a noisy
    "Task exception was never retrieved" traceback. The loop's own teardown has
    already released the underlying sockets, so simply dropping the reference and
    letting GC reclaim the objects is the correct, quiet cleanup.
    """
    for key, (loop, client) in list(_async_client_cache.items()):
        if loop.is_closed():
            del _async_client_cache[key]


def _get_or_create_async_client(model: str, project: str) -> "genai.Client":
    """Return a genai.Client bound to the CURRENT running event loop.

    Falls back to the shared sync cache if there is no running loop (which
    should not happen from async call sites, but keeps the helper safe).
    """
    location = get_location_for_model(model)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return _get_or_create_client(model, project)

    # Ensure this loop swallows the benign 'Event loop is closed' teardown noise
    # from previous loops' clients being finalized while this loop runs.
    _install_async_loop_exception_filter(loop)

    key = (location, project, id(loop))

    cached = _async_client_cache.get(key)
    # Identity check guards against id(loop) being reused by a new loop object.
    if cached is not None and cached[0] is loop and not loop.is_closed():
        return cached[1]

    with _async_client_cache_lock:
        _evict_dead_async_clients_locked()
        cached = _async_client_cache.get(key)
        if cached is not None and cached[0] is loop and not loop.is_closed():
            return cached[1]

        logging.info(f"🌍 Initializing per-loop async genai.Client - Location: {location}, Project: {project}")
        client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=types.HttpOptions(api_version='v1'),
        )
        _async_client_cache[key] = (loop, client)
        return client


def _close_cached_clients() -> None:
    """Close every cached client. Registered via atexit so process shutdown is clean."""
    with _client_cache_lock:
        for key, client in list(_client_cache.items()):
            try:
                client.close()
            except Exception as e:
                logging.debug(f"Error closing cached client {key}: {e}")
        _client_cache.clear()

    with _async_client_cache_lock:
        for key, (loop, client) in list(_async_client_cache.items()):
            # Only close while the bound loop is still alive. Closing an async
            # client on a dead loop schedules an aclose() Task on that closed
            # loop, raising "Event loop is closed". At process exit the loops are
            # virtually always already gone, so just drop the reference and let
            # GC reclaim them.
            if not loop.is_closed():
                try:
                    client.close()
                except Exception as e:
                    logging.debug(f"Error closing cached async client {key}: {e}")
        _async_client_cache.clear()


atexit.register(_close_cached_clients)


# ---------------------------------------------------------
# CLIENT CONTEXT MANAGERS
# ---------------------------------------------------------
# NOTE: These are kept as context managers for backwards compatibility with
# existing call sites, but they no longer create or close a client — they
# yield the cached singleton. The cached client is closed once at process
# exit via the atexit handler above.
@contextmanager
def get_genai_client(model: str, project: str = DEFAULT_PROJECT):
    """Yield the cached singleton Gemini client for (location, project)."""
    try:
        client = _get_or_create_client(model, project)
    except Exception as e:
        logging.error(f"🔴 Failed to initialize genai.Client: {e}")
        raise
    yield client


@asynccontextmanager
async def get_genai_client_async(model: str, project: str = DEFAULT_PROJECT):
    """Yield a Gemini client bound to the current event loop for (location, project)."""
    try:
        client = _get_or_create_async_client(model, project)
    except Exception as e:
        logging.error(f"🔴 Failed to initialize async genai.Client: {e}")
        raise
    yield client


# ---------------------------------------------------------
# CONFIGURATION BUILDER
# ---------------------------------------------------------
def build_generation_config(
    response_schema: Optional[Any] = None,
    system_instruction: Optional[str] = None,
    include_thoughts: bool = False,
    thinking_budget: int = 0,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    seed: Optional[int] = None,
) -> types.GenerateContentConfig:
    """
    Build generation configuration with safety settings.
    
    Args:
        response_schema: Schema for structured output
        include_thoughts: Enable thinking mode
        thinking_budget: Thinking token budget
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_output_tokens: Maximum output tokens
        seed: Random seed for reproducibility
        
    Returns:
        Configured GenerateContentConfig object
    """
    # Safety settings - block nothing for maximum flexibility
    safety_settings_list = [
        types.SafetySetting(
            category=category,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        )
        for category in [
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        ]
    ]
    
    config_dict = {
        "safety_settings": safety_settings_list,
        "thinking_config": types.ThinkingConfig(
            include_thoughts=include_thoughts,
            thinking_budget=thinking_budget
        ),
    }
    
    # Add optional parameters if provided
    if temperature is not None:
        config_dict["temperature"] = temperature
    if top_p is not None:
        config_dict["top_p"] = top_p
    if top_k is not None:
        config_dict["top_k"] = top_k
    if max_output_tokens is not None:
        config_dict["max_output_tokens"] = max_output_tokens
    if seed is not None:
        config_dict["seed"] = seed
    
    # Structured output configuration
    if response_schema:
        config_dict["response_mime_type"] = "application/json"
        config_dict["response_schema"] = response_schema
        
    if system_instruction:
        config_dict["system_instruction"] = system_instruction
    
    return types.GenerateContentConfig(**config_dict)


# ---------------------------------------------------------
# PARAMETER LOGGER
# ---------------------------------------------------------
def log_call_parameters(
    func_name: str,
    model: str,
    prompt: Optional[str] = None,
    text_input: Optional[str] = None,
    system_instruction: Optional[str] = None,
    contents: Optional[List] = None,
    response_schema: Optional[Any] = None,
    include_thoughts: bool = False,
    thinking_budget: int = 0,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    project: str = DEFAULT_PROJECT,
):
    """
    Log all parameters for debugging purposes.
    """
    logging.info("=" * 80)
    logging.info(f"🔧 FUNCTION CALL: {func_name}")
    logging.info("=" * 80)
    
    # Core parameters
    logging.info(f"📦 Model: {model}")
    logging.info(f"📦 Project: {project}")
    logging.info(f"📦 Location: {get_location_for_model(model)}")
    
    # Input parameters (commented out for privacy/log cleanliness)
    # if prompt is not None:
    #     prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
    #     logging.info(f"📝 Prompt: {prompt_preview}")
    
    # if text_input is not None:
    #     text_preview = text_input[:100] + "..." if len(text_input) > 100 else text_input
    #     logging.info(f"📝 Text Input: {text_preview}")
    
    if system_instruction is not None:
        logging.info(f"📝 System Instruction included: Yes ({len(system_instruction)} chars)")
    
    if contents is not None:
        content_types = [type(item).__name__ for item in contents]
        logging.info(f"📝 Contents: {len(contents)} items - Types: {content_types}")
    
    # Schema and thinking (response_schema commented out)
    # logging.info(f"🔧 Response Schema: {type(response_schema).__name__ if response_schema else 'None (Plain Text)'}")
    logging.info(f"🧠 Include Thoughts: {include_thoughts}")
    logging.info(f"🧠 Thinking Budget: {thinking_budget} tokens")
    
    # Sampling parameters
    logging.info("🎲 SAMPLING PARAMETERS:")
    logging.info(f"   ├─ Temperature: {temperature if temperature is not None else 'None (Gemini default ~1.0)'}")
    logging.info(f"   ├─ Top-P: {top_p if top_p is not None else 'None (Gemini default ~0.95)'}")
    logging.info(f"   ├─ Top-K: {top_k if top_k is not None else 'None (Gemini default ~40)'}")
    logging.info(f"   ├─ Max Output Tokens: {max_output_tokens if max_output_tokens is not None else 'None (Gemini default ~8192)'}")
    logging.info(f"   └─ Seed: {seed if seed is not None else 'None (Non-deterministic)'}")
    
    # Retry configuration
    logging.info(f"🔄 Max Retries: {MAX_RETRIES}")
    logging.info(f"🔄 Retry Delay: {RETRY_DELAY}s (exponential backoff)")
    
    logging.info("=" * 80)


# ---------------------------------------------------------
# RESPONSE VALIDATOR
# ---------------------------------------------------------
def validate_response(response: Any, model: str) -> Optional[str]:
    """
    Validate and extract text from Gemini response.
    
    Args:
        response: Gemini API response object
        model: Model name for logging
        
    Returns:
        Extracted text or None if invalid
    """
    if not response:
        logging.error("🔴 Response object is None")
        raise ValueError("Response object is None")
    
    if hasattr(response, 'text') and response.text:
        return response.text.strip()
    
    # Detailed error logging
    if hasattr(response, 'candidates') and response.candidates:
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
        
        # Check for safety blocks
        if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings is not None:
            blocked = [
                rating for rating in candidate.safety_ratings 
                if hasattr(rating, 'blocked') and rating.blocked
            ]
            if blocked:
                logging.error(
                    f"🔴 Response blocked by safety filters: {blocked}"
                )
                raise ValueError(f"Response blocked by safety filters: {blocked}")
        
        # ─── MAX_TOKENS: Try to extract partial text from candidates ───
        if 'MAX_TOKENS' in str(finish_reason):
            logging.warning(
                f"⚠️ Response truncated (MAX_TOKENS). Attempting to extract partial content..."
            )
            try:
                content = getattr(candidate, 'content', None)
                if content and hasattr(content, 'parts') and content.parts:
                    partial_texts = []
                    for part in content.parts:
                        part_text = getattr(part, 'text', None)
                        if part_text:
                            partial_texts.append(part_text)
                    if partial_texts:
                        partial_content = "".join(partial_texts).strip()
                        logging.warning(
                            f"⚠️ Extracted {len(partial_content)} chars of partial content (TRUNCATED — may be invalid JSON)"
                        )
                        # Save partial content to a debug file
                        try:
                            from pathlib import Path
                            from config.yaml_loader import PATHS
                            debug_dir = Path(PATHS.LOGS) / "debug_partial_responses"
                            debug_dir.mkdir(parents=True, exist_ok=True)
                            debug_file = debug_dir / f"partial_response_{int(time.time())}.txt"
                            with open(debug_file, "w", encoding="utf-8") as f:
                                f.write(f"MODEL: {model}\n")
                                f.write(f"FINISH_REASON: {finish_reason}\n")
                                f.write(f"CONTENT_LENGTH: {len(partial_content)}\n")
                                f.write(f"{'='*80}\n")
                                f.write(partial_content)
                            
                            # Upload to GCS
                            upload_file_to_gcs(str(debug_file), f"debug_partial_responses/{debug_file.name}")
                            
                            logging.warning(f"💾 Partial response saved to: {debug_file}")
                        except Exception as save_err:
                            logging.warning(f"⚠️ Could not save partial response to file: {save_err}")
                            
            except Exception as extract_err:
                logging.warning(f"⚠️ Failed to extract partial content: {extract_err}")
                
            # Raise ValueError OUTSIDE the try-except to trigger the retry decorator properly
            raise ValueError(f"Gemini call response truncated due to MAX_TOKENS finish reason for model: {model}")
        
        logging.warning(
            f"⚠️ No text in response. Finish reason: {finish_reason}"
        )
        raise ValueError(f"No text in response. Finish reason: {finish_reason}")
    else:
        logging.error("🔴 No candidates in response")
        raise ValueError("No candidates in response")


# ---------------------------------------------------------
# MAIN EXECUTION FUNCTIONS
# ---------------------------------------------------------
@retry_with_backoff(max_retries=MAX_RETRIES)
def execute_gemini_call(
    prompt: str,
    text_input: str,
    response_schema: Optional[Any] = None,
    system_instruction: Optional[str] = None,
    include_thoughts: bool = False,
    thinking_budget: int = 0,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    project: Optional[str] = None,
    page_number: Optional[int] = None,
) -> Optional[str]:
    """
    Execute a Gemini API call with text input (synchronous).
    
    Args:
        prompt: System/instruction prompt
        text_input: User text input
        response_schema: Schema for structured JSON output
        include_thoughts: Enable thinking mode
        thinking_budget: Token budget for thinking
        model: Model identifier
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_output_tokens: Maximum output tokens
        seed: Random seed for reproducibility
        project: GCP project ID
        
    Returns:
        Generated text or None on failure
    """
    project = project or ACTIVE_GCP_PROJECT
    
    # Log all parameters for debugging
    log_call_parameters(
        func_name="execute_gemini_call",
        model=model,
        prompt=prompt,
        text_input=text_input,
        system_instruction=system_instruction,
        response_schema=response_schema,
        include_thoughts=include_thoughts,
        thinking_budget=thinking_budget,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        seed=seed,
        project=project,
    )
    
    with get_genai_client(model, project) as client:
        generation_config = build_generation_config(
            response_schema=response_schema,
            system_instruction=system_instruction,
            include_thoughts=include_thoughts,
            thinking_budget=thinking_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )
        
        full_prompt = f"{prompt}\n\n{text_input}"
        
        logging.info(f"🔑 Executing Vertex AI call [Project: {project}] - Model: {model}")
        
        response = client.models.generate_content(
            model=model,
            contents=full_prompt,
            config=generation_config
        )
        
        # Log token usage
        log_token_usage(response, model)
        
        # Validate and return
        result = validate_response(response, model)
        if result:
            usage = getattr(response, 'usage_metadata', None)
            traffic = getattr(usage, 'traffic_type', 'UNKNOWN') if usage else 'UNKNOWN'
            logging.info(f"✅ Vertex AI call successful [Project: {project}] [Tier: {traffic}]")
        
        return result


@async_retry_with_backoff(max_retries=MAX_RETRIES)
async def execute_gemini_call_async(
    prompt: str,
    text_input: str,
    response_schema: Optional[Any] = None,
    system_instruction: Optional[str] = None,
    include_thoughts: bool = False,
    thinking_budget: int = 0,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    project: Optional[str] = None,
    page_number: Optional[int] = None,
) -> Optional[str]:
    """
    Execute a Gemini API call with text input (asynchronous).
    
    Args:
        prompt: System/instruction prompt
        text_input: User text input
        response_schema: Schema for structured JSON output
        include_thoughts: Enable thinking mode
        thinking_budget: Token budget for thinking
        model: Model identifier
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_output_tokens: Maximum output tokens
        seed: Random seed for reproducibility
        project: GCP project ID
        
    Returns:
        Generated text or None on failure
    """
    project = project or ACTIVE_GCP_PROJECT
    
    # Log all parameters for debugging
    log_call_parameters(
        func_name="execute_gemini_call_async",
        model=model,
        prompt=prompt,
        text_input=text_input,
        system_instruction=system_instruction,
        response_schema=response_schema,
        include_thoughts=include_thoughts,
        thinking_budget=thinking_budget,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        seed=seed,
        project=project,
    )
    
    async with get_genai_client_async(model, project) as client:
        generation_config = build_generation_config(
            response_schema=response_schema,
            system_instruction=system_instruction,
            include_thoughts=include_thoughts,
            thinking_budget=thinking_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )
        
        full_prompt = f"{prompt}\n\n{text_input}"
        
        logging.info(f"🔑 Executing async Vertex AI call [Project: {project}] - Model: {model}")
        
        response = await client.aio.models.generate_content(
            model=model,
            contents=full_prompt,
            config=generation_config
        )
        
        # Log token usage
        log_token_usage(response, model)
        
        # Validate and return
        result = validate_response(response, model)
        if result:
            usage = getattr(response, 'usage_metadata', None)
            traffic = getattr(usage, 'traffic_type', 'UNKNOWN') if usage else 'UNKNOWN'
            logging.info(f"✅ Async Vertex AI call successful [Project: {project}] [Tier: {traffic}]")
        
        return result


@retry_with_backoff(max_retries=MAX_RETRIES)
def execute_multimodal_gemini_call(
    contents: List[Union[str, PILImage.Image, types.Part]],
    response_schema: Optional[Any] = None,
    system_instruction: Optional[str] = None,
    include_thoughts: bool = False,
    thinking_budget: int = 0,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    project: Optional[str] = None,
    page_number: Optional[int] = None,
    context_label: Optional[str] = None,
) -> Optional[str]:
    """
    Execute a Gemini API call with multimodal input (synchronous).
    
    Args:
        contents: List of text strings and/or PIL Images
        response_schema: Schema for structured JSON output
        include_thoughts: Enable thinking mode
        thinking_budget: Token budget for thinking
        model: Model identifier
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_output_tokens: Maximum output tokens
        seed: Random seed for reproducibility
        project: GCP project ID
        
    Returns:
        Generated text or None on failure
    """
    project = project or ACTIVE_GCP_PROJECT
    
    # Log all parameters for debugging
    log_call_parameters(
        func_name="execute_multimodal_gemini_call",
        model=model,
        contents=contents,
        system_instruction=system_instruction,
        response_schema=response_schema,
        include_thoughts=include_thoughts,
        thinking_budget=thinking_budget,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        seed=seed,
        project=project,
    )
    
    with get_genai_client(model, project) as client:
        generation_config = build_generation_config(
            response_schema=response_schema,
            system_instruction=system_instruction,
            include_thoughts=include_thoughts,
            thinking_budget=thinking_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )
        
        # Validate and prepare contents
        sdk_contents = []
        for idx, item in enumerate(contents):
            if isinstance(item, (str, PILImage.Image, types.Part)):
                sdk_contents.append(item)
            else:
                logging.warning(
                    f"⚠️ Skipping unsupported content type at index {idx}: {type(item)}"
                )
        
        if not sdk_contents:
            logging.error("🔴 No valid contents provided")
            return None
        
        logging.info(
            f"🔑 Executing multimodal Vertex AI call [Project: {project}] - Model: {model}, "
            f"Items: {len(sdk_contents)}"
        )
        
        response = client.models.generate_content(
            model=model,
            contents=sdk_contents,
            config=generation_config
        )
        
        # Log token usage
        log_token_usage(response, model)
        
        # Validate and return
        result = validate_response(response, model)
        if result:
            usage = getattr(response, 'usage_metadata', None)
            traffic = getattr(usage, 'traffic_type', 'UNKNOWN') if usage else 'UNKNOWN'
            logging.info(f"✅ Multimodal Vertex AI call successful [Project: {project}] [Tier: {traffic}]")
        
        return result


@async_retry_with_backoff(max_retries=MAX_RETRIES)
async def execute_multimodal_gemini_call_async(
    contents: List[Union[str, PILImage.Image, types.Part]],
    response_schema: Optional[Any] = None,
    system_instruction: Optional[str] = None,
    include_thoughts: bool = False,
    thinking_budget: int = 0,
    model: str = DEFAULT_MODEL,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_output_tokens: Optional[int] = None,
    seed: Optional[int] = None,
    project: Optional[str] = None,
    page_number: Optional[int] = None,
    context_label: Optional[str] = None,
) -> Optional[str]:
    """
    Execute a Gemini API call with multimodal input (asynchronous).
    
    Args:
        contents: List of text strings and/or PIL Images
        response_schema: Schema for structured JSON output
        include_thoughts: Enable thinking mode
        thinking_budget: Token budget for thinking
        model: Model identifier
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        max_output_tokens: Maximum output tokens
        seed: Random seed for reproducibility
        project: GCP project ID
        
    Returns:
        Generated text or None on failure
    """
    project = project or ACTIVE_GCP_PROJECT
    
    # Log all parameters for debugging
    log_call_parameters(
        func_name="execute_multimodal_gemini_call_async",
        model=model,
        contents=contents,
        system_instruction=system_instruction,
        response_schema=response_schema,
        include_thoughts=include_thoughts,
        thinking_budget=thinking_budget,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        seed=seed,
        project=project,
    )
    
    async with get_genai_client_async(model, project) as client:
        generation_config = build_generation_config(
            response_schema=response_schema,
            system_instruction=system_instruction,
            include_thoughts=include_thoughts,
            thinking_budget=thinking_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )
        
        # Validate and prepare contents
        sdk_contents = []
        for idx, item in enumerate(contents):
            if isinstance(item, (str, PILImage.Image, types.Part)):
                sdk_contents.append(item)
            else:
                logging.warning(
                    f"⚠️ Skipping unsupported content type at index {idx}: {type(item)}"
                )
        
        if not sdk_contents:
            logging.error("🔴 No valid contents provided")
            return None
        
        logging.info(
            f"🔑 Executing async multimodal Vertex AI call [Project: {project}] - Model: {model}, "
            f"Items: {len(sdk_contents)}"
        )
        
        response = await client.aio.models.generate_content(
            model=model,
            contents=sdk_contents,
            config=generation_config
        )
        
        # Log token usage
        log_token_usage(response, model)
        
        # Validate and return
        result = validate_response(response, model)
        if result:
            usage = getattr(response, 'usage_metadata', None)
            traffic = getattr(usage, 'traffic_type', 'UNKNOWN') if usage else 'UNKNOWN'
            logging.info(f"✅ Async multimodal Vertex AI call successful [Project: {project}] [Tier: {traffic}]")
        
        return result
