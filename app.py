"""
app.py
──────
Infutrix Annotation Lab — Flask Web Server

Auto-detects mode on startup:
  • Full mode   : yolo_format_data/ present → Dataset Explorer + Dashboard
  • Person mode : images/ + labels/ + assignment.json present → Dashboard only

Run: python app.py
Open: http://localhost:5000
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from io import BytesIO
from pathlib import Path

from flask import (Flask, Response, abort, jsonify, redirect,
                   render_template, request, send_file, url_for)
from PIL import Image, ImageDraw

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TOKEN LOGGER (standalone, no GCS/DB)
# ─────────────────────────────────────────────────────────────────────────────
from token_usage_logger_local import log_token_usage, set_ocr_context

# ─────────────────────────────────────────────────────────────────────────────
# VERTEX AI / GEMINI
# ─────────────────────────────────────────────────────────────────────────────
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
YOLO_DIR        = BASE_DIR / 'yolo_format_data'
DIST_DIR        = BASE_DIR / 'distributions'
ASSIGNED_DATA_DIR = BASE_DIR / 'assigned_data'
OCR_TRAIN_DIR   = BASE_DIR / 'ocr_training'
ANNOTATIONS_FILE= OCR_TRAIN_DIR / 'annotations.json'
SETTINGS_FILE   = BASE_DIR / 'lab_settings.json'
LOG_DIR         = BASE_DIR / 'logs'

LOG_DIR.mkdir(exist_ok=True)
ASSIGNED_DATA_DIR.mkdir(exist_ok=True)
OCR_TRAIN_DIR.mkdir(exist_ok=True)
(OCR_TRAIN_DIR / 'images').mkdir(exist_ok=True)

SUPPORTED_EXTS  = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'app.log', encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256 MB


# ─────────────────────────────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    'model':      'gemini-3.1-flash-lite',
    'project':    'gradesmith-demo',
    'batch_size': 10,
    'concurrency': 10,
    'annotator':  'Annotator',
}

def load_settings() -> dict:
    settings = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, encoding='utf-8') as f:
                settings.update(json.load(f))
        except Exception:
            pass
            
    # Auto-detect annotator from assignment.json if in person mode
    assigned_folder = get_assigned_folder()
    if assigned_folder:
        try:
            with open(assigned_folder / 'assignment.json', encoding='utf-8') as f:
                assignment = json.load(f)
                if 'annotator' in assignment:
                    settings['annotator'] = assignment['annotator']
        except Exception:
            pass
            
    return settings

def save_settings(data: dict):
    merged = {**load_settings(), **data}
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MODE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def get_assigned_folder() -> Path | None:
    """Scan assigned_data/ for the first subfolder containing assignment.json."""
    if ASSIGNED_DATA_DIR.exists():
        for sub in ASSIGNED_DATA_DIR.iterdir():
            if sub.is_dir() and (sub / 'assignment.json').exists():
                return sub
    return None


def detect_mode() -> str:
    """
    Returns:
      'full'   — yolo_format_data/ present
      'person' — assigned_data/ <Annotator>_<Date> / assignment.json present
      'none'   — nothing found
    """
    if YOLO_DIR.exists() and any((YOLO_DIR / s / 'images').exists() for s in ['train', 'test', 'valid']):
        return 'full'
    if get_assigned_folder() is not None:
        return 'person'
    return 'none'


# ─────────────────────────────────────────────────────────────────────────────
# IMAGE CATALOGUE
# ─────────────────────────────────────────────────────────────────────────────
def build_image_catalogue() -> list[dict]:
    """
    Scan dataset and return sorted list of image records:
      { id, filename, split, image_path, label_path, annotation_count }
    """
    mode   = detect_mode()
    items  = []
    splits = ['train', 'test', 'valid']

    if mode == 'full':
        for split in splits:
            img_dir   = YOLO_DIR / split / 'images'
            label_dir = YOLO_DIR / split / 'labels'
            if not img_dir.exists():
                continue
            for img_file in sorted(img_dir.iterdir()):
                if img_file.suffix.lower() not in SUPPORTED_EXTS:
                    continue
                label_file = label_dir / (img_file.stem + '.txt')
                n_ann = 0
                if label_file.exists():
                    try:
                        lines = label_file.read_text().strip().split('\n')
                        n_ann = sum(1 for l in lines if l.strip())
                    except Exception:
                        pass
                items.append({
                    'id':               img_file.stem,
                    'filename':         img_file.name,
                    'split':            split,
                    'image_path':       str(img_file),
                    'label_path':       str(label_file) if label_file.exists() else None,
                    'annotation_count': n_ann,
                })

    elif mode == 'person':
        assigned_dir = get_assigned_folder()
        if not assigned_dir:
            return []
            
        img_dir   = assigned_dir / 'images'
        label_dir = assigned_dir / 'labels'
        try:
            assignment = json.loads((assigned_dir / 'assignment.json').read_text())
        except Exception:
            assignment = {}
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in SUPPORTED_EXTS:
                continue
            label_file = label_dir / (img_file.stem + '.txt')
            n_ann = 0
            if label_file.exists():
                try:
                    lines = label_file.read_text().strip().split('\n')
                    n_ann = sum(1 for l in lines if l.strip())
                except Exception:
                    pass
            items.append({
                'id':               img_file.stem,
                'filename':         img_file.name,
                'split':            'assigned',
                'image_path':       str(img_file),
                'label_path':       str(label_file) if label_file.exists() else None,
                'annotation_count': n_ann,
            })

    return items


def get_image_record(filename: str) -> dict | None:
    for item in build_image_catalogue():
        if item['filename'] == filename:
            return item
    return None


# ─────────────────────────────────────────────────────────────────────────────
# ANNOTATIONS (GT) — read / write
# ─────────────────────────────────────────────────────────────────────────────
def load_annotations() -> dict:
    """Load annotations.json → dict of {filename: {ocr_text, status, ...}}"""
    if ANNOTATIONS_FILE.exists():
        try:
            with open(ANNOTATIONS_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_annotations(data: dict):
    with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_image_ocr_status(filename: str) -> dict:
    """Return OCR record for one image, or a default pending record."""
    ann = load_annotations()
    return ann.get(filename, {
        'ocr_text':         '',
        'status':           'pending',
        'ocr_model':        '',
        'ocr_timestamp':    '',
        'annotator':        '',
        'contains_diagram': False,
    })


# ─────────────────────────────────────────────────────────────────────────────
# YOLO POLYGON PARSING
# ─────────────────────────────────────────────────────────────────────────────
def parse_label_file(label_path: str) -> list[list[tuple[float, float]]]:
    """
    Parse a YOLO segmentation .txt file.
    Returns list of polygons, each polygon is list of (x_norm, y_norm).
    """
    polygons = []
    try:
        with open(label_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                # parts[0] = class_id, rest = x1 y1 x2 y2 ...
                coords = list(map(float, parts[1:]))
                poly   = [(coords[i], coords[i+1]) for i in range(0, len(coords) - 1, 2)]
                if len(poly) >= 3:
                    polygons.append(poly)
    except Exception as e:
        log.warning(f"Failed to parse label file {label_path}: {e}")
    return polygons


# ─────────────────────────────────────────────────────────────────────────────
# OCR — image masking + Gemini call
# ─────────────────────────────────────────────────────────────────────────────
OCR_PROMPT = """You are an expert OCR assistant. Extract ALL visible text from this image.
The black-filled regions are intentionally hidden — do not mention them or describe them.

Formatting rules:
1. If text appears in a table, render it as a Markdown table (pipes and dashes).
2. Preserve all line breaks exactly as they appear in the source document.
3. Preserve indentation and spatial layout as much as possible.
4. Output ONLY the extracted text — no preamble, no explanation, no commentary."""


def mask_image_for_ocr(image_path: str, label_path: str | None) -> bytes:
    """
    Load image at full resolution, black-fill all annotated polygon regions.
    Returns raw JPEG bytes (quality=95, no downscaling).
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    if label_path:
        draw     = ImageDraw.Draw(img)
        polygons = parse_label_file(label_path)
        for poly in polygons:
            # Convert normalised → pixel
            pixel_poly = [(int(x * w), int(y * h)) for x, y in poly]
            draw.polygon(pixel_poly, fill=(0, 0, 0))

    buf = BytesIO()
    img.save(buf, format='JPEG', quality=95, subsampling=0)  # subsampling=0 = best quality
    return buf.getvalue()
# Client cache to prevent re-authenticating and rebuilding the client on every request
_client_cache = {}
_client_lock = threading.Lock()


def _make_genai_client(model: str, project: str):
    """Create a Vertex AI genai.Client for the given model."""
    location_map = {
        'gemini-2.5-flash-lite':         'asia-south1',
        'gemini-2.5-flash':              'asia-south1',
        'gemini-2.5-pro':                'asia-south1',
        'gemini-3.1-flash-lite-preview': 'global',
        'gemini-3.1-flash-lite':         'global',
    }
    location = next((v for k, v in location_map.items() if k in model.lower()), 'asia-south1')
    
    cache_key = (project, location)
    with _client_lock:
        if cache_key not in _client_cache:
            _client_cache[cache_key] = genai.Client(
                vertexai=True,
                project=project,
                location=location,
                http_options=types.HttpOptions(api_version='v1'),
            )
        return _client_cache[cache_key]


def _call_gemini_ocr(image_bytes: bytes, model: str, project: str, annotator: str, filename: str) -> str:
    """
    Send masked image bytes to Gemini and return extracted text.
    Uses set_ocr_context so token usage is logged with image name.
    Includes 3 retries with exponential backoff on transient network errors (e.g. RemoteDisconnected).
    """
    import time
    set_ocr_context(annotator=annotator, image_name=filename, session_id='annotation_lab')

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            client = _make_genai_client(model, project)

            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_text(text=OCR_PROMPT),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type='image/jpeg',
                    ),
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=8192,
                    safety_settings=[
                        types.SafetySetting(
                            category=cat,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        )
                        for cat in [
                            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        ]
                    ],
                ),
            )

            log_token_usage(response, model)

            if hasattr(response, 'text') and response.text:
                return response.text.strip()
            raise ValueError("Empty response from Gemini")

        except Exception as e:
            log.warning(f"⚠️ Attempt {attempt + 1}/{max_attempts} failed for {filename}: {e}")
            if attempt == max_attempts - 1:
                # Raise the last exception if we ran out of retries
                raise e
            
            # Clear client cache on connection errors to force fresh auth and socket next time
            location_map = {
                'gemini-2.5-flash-lite':         'asia-south1',
                'gemini-2.5-flash':              'asia-south1',
                'gemini-2.5-pro':                'asia-south1',
                'gemini-3.1-flash-lite-preview': 'global',
                'gemini-3.1-flash-lite':         'global',
            }
            loc = next((v for k, v in location_map.items() if k in model.lower()), 'asia-south1')
            cache_key = (project, loc)
            with _client_lock:
                if cache_key in _client_cache:
                    del _client_cache[cache_key]
                    
            # Wait before retrying (exponential backoff)
            time.sleep(1.0 * (attempt + 1))


# ─────────────────────────────────────────────────────────────────────────────
# OCR BATCH JOB STATE (in-memory, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────
_ocr_jobs: dict[str, dict] = {}   # job_id → job state
_ocr_lock = threading.Lock()


def _get_or_create_job(job_id: str) -> dict:
    with _ocr_lock:
        if job_id not in _ocr_jobs:
            _ocr_jobs[job_id] = {
                'status':    'idle',
                'total':     0,
                'done':      0,
                'running':   0,
                'failed':    0,
                'queued':    0,
                'results':   {},     # filename → {'status': ..., 'text': ..., 'error': ...}
                'started_at': None,
            }
        return _ocr_jobs[job_id]


ACTIVE_JOB_ID = 'main'   # single global job for simplicity


def _run_ocr_batch(filenames: list[str], model: str, project: str,
                   annotator: str, concurrency: int, job_id: str):
    """
    Run OCR on a list of filenames concurrently.
    Updates job state in _ocr_jobs as each image is processed.
    """
    job = _get_or_create_job(job_id)
    catalogue = {item['filename']: item for item in build_image_catalogue()}

    def process_one(filename: str):
        with _ocr_lock:
            job['running'] += 1
            job['queued']  -= 1
            job['results'][filename] = {'status': 'running', 'text': '', 'error': ''}

        result_status = 'failed'
        result_text   = ''
        result_error  = ''

        try:
            record = catalogue.get(filename)
            if not record:
                raise FileNotFoundError(f"Image not in catalogue: {filename}")

            img_bytes = mask_image_for_ocr(record['image_path'], record['label_path'])
            text      = _call_gemini_ocr(img_bytes, model, project, annotator, filename)

            # Save to annotations.json immediately
            ann = load_annotations()
            existing = ann.get(filename, {})
            ann[filename] = {
                'ocr_text':      text,
                'ocr_raw_text':  text,           # original AI output — never overwritten by user edits
                'status':        'pending',       # pending = OCR done, awaiting human review
                'ocr_model':     model,
                'ocr_timestamp': datetime.now().isoformat(timespec='seconds'),
                'annotator':     annotator,
                'contains_diagram': existing.get('contains_diagram', False),
            }
            save_annotations(ann)

            # Copy image to ocr_training/images/ if not already there
            dest = OCR_TRAIN_DIR / 'images' / filename
            if not dest.exists():
                shutil.copy2(record['image_path'], dest)

            result_status = 'done'
            result_text   = text

        except Exception as e:
            result_error = str(e)
            log.error(f"OCR failed for {filename}: {e}")

        with _ocr_lock:
            job['running'] -= 1
            job['done']    += (1 if result_status == 'done' else 0)
            job['failed']  += (1 if result_status == 'failed' else 0)
            job['results'][filename] = {
                'status': result_status,
                'text':   result_text,
                'error':  result_error,
            }

    with _ocr_lock:
        job['status']     = 'running'
        job['total']      = len(filenames)
        job['done']       = 0
        job['running']    = 0
        job['failed']     = 0
        job['queued']     = len(filenames)
        job['results']    = {fn: {'status': 'queued', 'text': '', 'error': ''} for fn in filenames}
        job['started_at'] = datetime.now().isoformat(timespec='seconds')

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        # consume the map iterator to ensure all futures execute and exceptions are raised if any
        list(executor.map(process_one, filenames))

    with _ocr_lock:
        job['status'] = 'completed'
    log.info(f"OCR batch completed: {job['done']} done, {job['failed']} failed")


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — PAGES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    mode = detect_mode()
    if mode == 'person':
        return redirect(url_for('dashboard'))
    if mode == 'none':
        return render_template('index.html', mode='none', images=[],
                               stats={'total': 0, 'validated': 0, 'pending': 0,
                                      'done_ocr': 0, 'failed': 0, 'splits': {}})

    images    = build_image_catalogue()
    ann       = load_annotations()
    settings  = load_settings()

    for img in images:
        rec = ann.get(img['filename'], {})
        img['ocr_status'] = rec.get('status', 'pending')
        img['contains_diagram'] = rec.get('contains_diagram', False)

    splits  = {}
    for img in images:
        splits[img['split']] = splits.get(img['split'], 0) + 1

    stats = {
        'total':     len(images),
        'validated': sum(1 for r in ann.values() if isinstance(r, dict) and r.get('status') == 'validated'),
        'done_ocr':  sum(1 for r in ann.values() if isinstance(r, dict) and r.get('status') in ('pending', 'validated')),
        'pending':   sum(1 for img in images if ann.get(img['filename'], {}).get('status', 'pending') == 'pending'),
        'failed':    0,
        'splits':    splits,
    }

    return render_template('index.html', mode='full', images=images, stats=stats, settings=settings)


@app.route('/dashboard')
def dashboard():
    settings = load_settings()
    images   = build_image_catalogue()
    ann      = load_annotations()

    for img in images:
        rec = ann.get(img['filename'], {})
        img['ocr_status'] = rec.get('status', 'pending') if isinstance(rec, dict) else 'pending'
        img['ocr_text']   = rec.get('ocr_text', '')     if isinstance(rec, dict) else ''
        img['contains_diagram'] = rec.get('contains_diagram', False) if isinstance(rec, dict) else False

    stats = {
        'total':     len(images),
        'validated': sum(1 for img in images if img['ocr_status'] == 'validated'),
        'pending':   sum(1 for img in images if img['ocr_status'] == 'pending'),
        'skipped':   sum(1 for img in images if img['ocr_status'] == 'skipped'),
        # OCR bar: images where AI has actually run OCR (has ocr_timestamp)
        'ocr_done':  sum(1 for fn, rec in ann.items() if isinstance(rec, dict) and rec.get('ocr_timestamp')),
        # Reviewed bar: images a human has explicitly saved (has saved_timestamp)
        'reviewed':  sum(1 for fn, rec in ann.items() if isinstance(rec, dict) and rec.get('saved_timestamp')),
    }

    return render_template('dashboard.html',
                           images=images,
                           stats=stats,
                           settings=settings,
                           job=_get_or_create_job(ACTIVE_JOB_ID))


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/api/images')
def api_images():
    images = build_image_catalogue()
    ann    = load_annotations()
    for img in images:
        rec = ann.get(img['filename'], {}) if isinstance(ann.get(img['filename']), dict) else {}
        img['ocr_status']      = rec.get('status', 'pending')
        img['ocr_text']        = rec.get('ocr_text', '')
        img['ocr_timestamp']   = rec.get('ocr_timestamp', '')   # set only when AI ran OCR
        img['saved_timestamp'] = rec.get('saved_timestamp', '') # set only when human saved
        img['human_edited']    = rec.get('human_edited', False) # True if text was changed vs AI output
        img['contains_diagram'] = rec.get('contains_diagram', False)
    return jsonify(images)


@app.route('/api/image/<path:filename>')
def api_image(filename):
    """Serve an image file at full quality."""
    record = get_image_record(filename)
    if not record:
        abort(404)
    return send_file(record['image_path'], mimetype='image/jpeg')


@app.route('/api/annotations/<path:filename>')
def api_annotations(filename):
    """Return parsed polygon coordinates for an image."""
    record = get_image_record(filename)
    if not record:
        return jsonify([])
    if not record.get('label_path'):
        return jsonify([])
    polygons = parse_label_file(record['label_path'])
    return jsonify(polygons)


@app.route('/api/stats')
def api_stats():
    images = build_image_catalogue()
    ann    = load_annotations()
    splits = {}
    for img in images:
        splits[img['split']] = splits.get(img['split'], 0) + 1

    return jsonify({
        'total':     len(images),
        'validated': sum(1 for r in ann.values() if isinstance(r, dict) and r.get('status') == 'validated'),
        'done_ocr':  sum(1 for r in ann.values() if isinstance(r, dict) and r.get('status') in ('pending', 'validated')),
        'pending':   sum(1 for img in images if ann.get(img['filename'], {}).get('status', 'pending') not in ('validated', 'skipped')),
        'splits':    splits,
    })


@app.route('/api/ocr/start', methods=['POST'])
def api_ocr_start():
    """Start an OCR batch job."""
    data       = request.get_json() or {}
    settings   = load_settings()

    model      = data.get('model',       settings['model'])
    project    = data.get('project',     settings['project'])
    concurrency= int(data.get('concurrency', settings['concurrency']))
    annotator  = data.get('annotator',   settings['annotator'])
    retry_failed = data.get('retry_failed', False)
    target_filename = data.get('filename', None)

    # Decide which images to process
    images = build_image_catalogue()
    ann    = load_annotations()

    if target_filename:
        filenames = [target_filename]
    elif retry_failed:
        job = _get_or_create_job(ACTIVE_JOB_ID)
        filenames = [fn for fn, r in job['results'].items() if r['status'] == 'failed']
    else:
        filenames = [
            img['filename'] for img in images
            if ann.get(img['filename'], {}).get('status', 'pending') not in ('validated', 'skipped')
            and _get_or_create_job(ACTIVE_JOB_ID)['results'].get(img['filename'], {}).get('status', '') not in ('running', 'queued')
        ]

    if not filenames:
        return jsonify({'message': 'No pending images to process.', 'count': 0})

    # Run in background thread
    t = threading.Thread(
        target=_run_ocr_batch,
        args=(filenames, model, project, annotator, concurrency, ACTIVE_JOB_ID),
        daemon=True,
    )
    t.start()

    return jsonify({'message': 'OCR batch started', 'count': len(filenames), 'job_id': ACTIVE_JOB_ID})


@app.route('/api/ocr/status')
def api_ocr_status():
    """Return current OCR batch status (polled by frontend every second)."""
    job = _get_or_create_job(ACTIVE_JOB_ID)
    with _ocr_lock:
        return jsonify({
            'status':   job['status'],
            'total':    job['total'],
            'done':     job['done'],
            'running':  job['running'],
            'failed':   job['failed'],
            'queued':   job['queued'],
            'results':  job['results'],
        })


@app.route('/api/save_gt', methods=['POST'])
def api_save_gt():
    """Save / update one image's OCR text and status."""
    data = request.get_json() or {}
    filename  = data.get('filename', '').strip()
    ocr_text  = data.get('ocr_text', '')
    status    = data.get('status', 'pending')   # 'pending' | 'validated' | 'skipped'
    settings  = load_settings()
    annotator = data.get('annotator', settings.get('annotator', 'N/A'))

    if not filename:
        return jsonify({'error': 'filename required'}), 400

    ann = load_annotations()
    existing = ann.get(filename, {})

    # Detect if human edited the text vs original OCR output
    ocr_raw = existing.get('ocr_raw_text', '')  # set by AI, never changed by user
    if ocr_raw:
        human_edited = (ocr_text.strip() != ocr_raw.strip())
    else:
        # No OCR was ever run — any text here was manually entered
        human_edited = bool(ocr_text.strip())

    contains_diagram = bool(data.get('contains_diagram', False))

    ann[filename] = {
        'ocr_text':         ocr_text,
        'ocr_raw_text':     existing.get('ocr_raw_text', ''),  # preserve original AI text
        'status':           status,
        'ocr_model':        existing.get('ocr_model', ''),
        'ocr_timestamp':    existing.get('ocr_timestamp', ''),
        'saved_timestamp':  datetime.now().isoformat(timespec='seconds'),
        'human_edited':     human_edited,
        'annotator':        annotator,
        'contains_diagram': contains_diagram,
    }
    save_annotations(ann)

    # Ensure image is in ocr_training/images/
    record = get_image_record(filename)
    if record:
        dest = OCR_TRAIN_DIR / 'images' / filename
        if not dest.exists():
            shutil.copy2(record['image_path'], dest)

    return jsonify({'ok': True, 'filename': filename, 'status': status})


@app.route('/api/delete_image', methods=['DELETE'])
def api_delete_image():
    """
    Delete an image + its label + its annotation entry.
    Requires body: {"filename": "...", "confirm": "DELETE"}
    """
    data    = request.get_json() or {}
    filename = data.get('filename', '').strip()
    confirm  = data.get('confirm', '').strip()

    if not filename:
        return jsonify({'error': 'filename required'}), 400
    if confirm != 'DELETE':
        return jsonify({'error': 'You must pass confirm="DELETE" to proceed.'}), 400

    record = get_image_record(filename)
    deleted = []

    if record:
        # Delete image
        img_p = Path(record['image_path'])
        if img_p.exists():
            img_p.unlink()
            deleted.append(str(img_p))
        # Delete label
        if record.get('label_path'):
            lbl_p = Path(record['label_path'])
            if lbl_p.exists():
                lbl_p.unlink()
                deleted.append(str(lbl_p))

    # Delete from ocr_training/images/
    ocr_img = OCR_TRAIN_DIR / 'images' / filename
    if ocr_img.exists():
        ocr_img.unlink()
        deleted.append(str(ocr_img))

    # Remove from annotations.json
    ann = load_annotations()
    if filename in ann:
        del ann[filename]
        save_annotations(ann)

    # Remove from in-memory OCR job
    with _ocr_lock:
        job = _ocr_jobs.get(ACTIVE_JOB_ID, {})
        if filename in job.get('results', {}):
            del job['results'][filename]

    log.info(f"Deleted: {filename} — files removed: {deleted}")
    return jsonify({'ok': True, 'filename': filename, 'deleted_files': deleted})


@app.route('/api/settings', methods=['GET', 'POST'])
def api_settings():
    if request.method == 'GET':
        return jsonify(load_settings())
    data = request.get_json() or {}
    save_settings(data)
    return jsonify({'ok': True, 'settings': load_settings()})


@app.route('/api/export')
def api_export():
    """Download annotations.json as a file."""
    if not ANNOTATIONS_FILE.exists():
        return jsonify({'error': 'No annotations yet'}), 404
    return send_file(
        str(ANNOTATIONS_FILE),
        as_attachment=True,
        download_name=f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )


@app.route('/api/image_ocr/<path:filename>')
def api_image_ocr(filename):
    """Return saved OCR data for one image."""
    return jsonify(get_image_ocr_status(filename))


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    mode = detect_mode()
    log.info(f"🧬 Infutrix Annotation Lab starting in mode: {mode}")
    log.info(f"   Base dir     : {BASE_DIR}")
    log.info(f"   Dataset      : {YOLO_DIR}")
    log.info(f"   OCR output   : {OCR_TRAIN_DIR}")
    log.info(f"   Token log    : {LOG_DIR / 'annotation_lab_token_usage.csv'}")
    print(f"\n  🌐  Open: http://localhost:5000\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
