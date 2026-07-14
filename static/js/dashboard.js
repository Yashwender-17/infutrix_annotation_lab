/**
 * dashboard.js
 * ─────────────
 * Annotation Lab — Annotator Dashboard Logic
 *
 * Responsibilities:
 *  • Image navigation (prev / next / thumbnail click)
 *  • Auto-save when navigating away
 *  • Canvas rendering with YOLO polygon overlay
 *  • OCR batch launch + live progress polling
 *  • Delete with "DELETE" confirmation
 *  • Settings modal
 *  • Keyboard shortcuts (Enter = save+next, ←/→ = nav)
 */

'use strict';

import { drawAnnotatedImage, fetchPolygons } from './viewer.js';

// ─────────────────────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────────────────────
let _images      = [];    // full list from server
let _curIndex    = 0;     // current position in _images
let _showMask    = true;  // toggle overlay
let _settings    = {};    // server settings
let _pollTimer   = null;  // OCR status poll interval
let _saveTimer   = null;  // debounce auto-save
let _dirty       = false; // unsaved changes in textarea
let _curImgOcrRunning = false; // true while OCR is running for the current image

// ─────────────────────────────────────────────────────────────────────────────
// DOM REFS
// ─────────────────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

const mainCanvas      = $('mainCanvas');
const ocrTextarea     = $('ocrTextarea');
const jumpInput       = $('jumpInput');
const navTotal        = $('navTotal');
const btnJump         = $('btnJump');
const btnThemeToggle  = $('btnThemeToggle');
const chkContainsDiagram = $('chkContainsDiagram');
const diagramBadge    = $('diagramBadge');
const thumbStrip      = $('thumbStrip');
const filenameLabel   = $('filenameLabel');
const splitLabel      = $('splitLabel');
const annotCountLabel = $('annotCountLabel');
const ocrStatusBadge  = $('ocrStatusBadge');

// OCR progress counters
const cntDone     = $('cntDone');
const cntRunning  = $('cntRunning');
const cntFailed   = $('cntFailed');
const cntQueued   = $('cntQueued');
const progressBar = $('ocrProgressBar');

// Buttons
const btnPrev       = $('btnPrev');
const btnNext       = $('btnNext');
const btnToggleMask = $('btnToggleMask');
const btnRunOcr     = $('btnRunOcr');
const btnRunOcrCurr = $('btnRunOcrCurrent');
const btnRetry      = $('btnRetry');
const btnSave       = $('btnSave');
const btnDelete     = $('btnDelete');
const btnSettings   = $('btnSettings');
const btnExport     = $('btnExport');

// Modals
const deleteModal    = $('deleteModal');
const settingsModal  = $('settingsModal');

// Status radios
const statusRadios = document.querySelectorAll('.status-radio-item');

// ─────────────────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────────────────
async function init() {
  _settings = await fetchJSON('/api/settings');
  await loadImages();
  buildThumbStrip();
  await navigateTo(0, false);
  attachListeners();
  // Check once on boot to see if a background job is already running
  await pollOcrStatus();
}

// ─────────────────────────────────────────────────────────────────────────────
// DATA
// ─────────────────────────────────────────────────────────────────────────────
async function loadImages() {
  _images = await fetchJSON('/api/images');
}

function currentImage() {
  return _images[_curIndex] || null;
}

// ─────────────────────────────────────────────────────────────────────────────
// NAVIGATION
// ─────────────────────────────────────────────────────────────────────────────
async function navigateTo(index, autoSavePrev = true) {
  if (autoSavePrev) {
    await autoSave();
  }

  if (index < 0) index = 0;
  if (index >= _images.length) index = _images.length - 1;
  _curIndex = index;

  // Reset OCR-running flag so a previous image's state doesn't bleed over
  _curImgOcrRunning = false;

  const img = currentImage();
  if (!img) return;

  // Update labels
  filenameLabel.textContent   = img.filename;
  splitLabel.textContent      = img.split || '—';
  annotCountLabel.textContent = `${img.annotation_count || 0} polygon(s)`;
  if (jumpInput) jumpInput.value = _curIndex + 1;
  if (navTotal) navTotal.textContent = _images.length;

  // Update strip highlight
  document.querySelectorAll('.strip-thumb').forEach((el, i) => {
    el.classList.toggle('active', i === _curIndex);
  });
  // Scroll strip thumb into view
  const activeThumb = thumbStrip.querySelector('.strip-thumb.active');
  if (activeThumb) activeThumb.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });

  // Load OCR data for this image
  const ocrData = await fetchJSON(`/api/image_ocr/${img.filename}`);
  ocrTextarea.value = ocrData.ocr_text || '';

  // Set status radio
  setStatusRadio(ocrData.status || 'pending');

  if (chkContainsDiagram) {
    chkContainsDiagram.checked = !!ocrData.contains_diagram;
  }
  if (diagramBadge) {
    diagramBadge.style.display = ocrData.contains_diagram ? 'inline-flex' : 'none';
  }

  // Update OCR status badge
  updateOcrBadge(ocrData.status || 'pending');

  // Update edited badge + last saved
  updateEditedBadge(ocrData);
  _dirty = false;
  ocrTextarea.classList.remove('unsaved');
  const unsavedEl = $('unsavedIndicator');
  if (unsavedEl) unsavedEl.style.display = 'none';

  // Draw canvas
  await renderMainCanvas(img);

  // Nav button states
  btnPrev.disabled = (_curIndex === 0);
  btnNext.disabled = (_curIndex === _images.length - 1);
}

function doJump() {
  if (!jumpInput) return;
  const val = parseInt(jumpInput.value, 10);
  if (!isNaN(val)) {
    navigateTo(val - 1);
  }
}



async function renderMainCanvas(img) {
  if (!mainCanvas) return;
  try {
    const polygons = await fetchPolygons(img.filename);
    await drawAnnotatedImage(mainCanvas, `/api/image/${img.filename}`, polygons, {
      showMask:  _showMask,
      lineWidth: 2,
    });
  } catch (e) {
    console.error('Canvas render failed:', e);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SAVE STATUS UI
// ─────────────────────────────────────────────────────────────────────────────
const SAVE_STYLES = {
  idle:   { bg: 'var(--bg-base)',            border: 'var(--border)',       icon: '💾', text: 'var(--text-muted)' },
  dirty:  { bg: 'rgba(217,119,6,0.07)',      border: 'var(--amber)',        icon: '⚠️',  text: 'var(--amber)'     },
  saving: { bg: 'rgba(37,99,235,0.07)',      border: 'var(--blue)',         icon: '🔄', text: 'var(--blue)'      },
  saved:  { bg: 'rgba(5,150,105,0.08)',      border: 'var(--teal)',         icon: '✅', text: 'var(--teal)'      },
  error:  { bg: 'rgba(220,38,38,0.08)',      border: 'var(--red)',          icon: '❌', text: 'var(--red)'       },
};

let _saveCountdownTimer = null;
let _saveCountdownSecs  = 0;

function setSaveState(state, message = '', timeLabel = '') {
  const bar      = $('saveStatusBar');
  const iconEl   = $('saveStatusIcon');
  const textEl   = $('saveStatusText');
  const timeEl   = $('saveStatusTime');
  if (!bar) return;

  const s = SAVE_STYLES[state] || SAVE_STYLES.idle;
  bar.style.background   = s.bg;
  bar.style.borderColor  = s.border;
  if (iconEl) { iconEl.textContent = s.icon; }
  if (textEl) { textEl.textContent = message || ''; textEl.style.color = s.text; }
  if (timeEl) { timeEl.textContent = timeLabel; timeEl.style.color = s.text; }

  // Also keep the textarea border in sync
  if (ocrTextarea) {
    ocrTextarea.classList.toggle('unsaved', state === 'dirty');
  }
}

function startSaveCountdown(secs) {
  clearInterval(_saveCountdownTimer);
  _saveCountdownSecs = secs;
  setSaveState('dirty', 'Unsaved changes', `auto-save in ${_saveCountdownSecs}s`);

  _saveCountdownTimer = setInterval(() => {
    _saveCountdownSecs--;
    if (_saveCountdownSecs <= 0) {
      clearInterval(_saveCountdownTimer);
      setSaveState('saving', 'Saving…', '');
    } else {
      setSaveState('dirty', 'Unsaved changes', `auto-save in ${_saveCountdownSecs}s`);
    }
  }, 1000);
}

function stopSaveCountdown() {
  clearInterval(_saveCountdownTimer);
  _saveCountdownTimer = null;
}

// ─────────────────────────────────────────────────────────────────────────────
// AUTO-SAVE
// ─────────────────────────────────────────────────────────────────────────────
async function autoSave() {
  const img = currentImage();
  if (!img) return;

  stopSaveCountdown();
  setSaveState('saving', 'Saving…', '');

  const text   = ocrTextarea.value;
  const status = getStatusRadioValue();
  const containsDiagram = chkContainsDiagram ? chkContainsDiagram.checked : false;

  try {
    const res = await fetch('/api/save_gt', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        filename:  img.filename,
        ocr_text:  text,
        status:    status,
        annotator: _settings.annotator || 'Annotator',
        contains_diagram: containsDiagram,
      }),
    });

    // Update in-memory record so bars update live
    _images[_curIndex].ocr_status       = status;
    _images[_curIndex].ocr_text         = text;
    _images[_curIndex].contains_diagram = containsDiagram;
    _images[_curIndex].saved_timestamp  = new Date().toISOString(); // mark as reviewed
    updateStripDot(_curIndex, status);
    updateStatsBar();

    // Clear dirty / unsaved state
    _dirty = false;
    ocrTextarea.classList.remove('unsaved');
    const unsavedEl = $('unsavedIndicator');
    if (unsavedEl) unsavedEl.style.display = 'none';

    // Refresh edited badge + last saved time (server computed human_edited)
    const fresh = await fetchJSON(`/api/image_ocr/${img.filename}`);
    updateEditedBadge(fresh);
    // Also update in-memory human_edited
    _images[_curIndex].human_edited = fresh.human_edited || false;
  } catch (e) {
    console.error('Auto-save failed:', e);
  }
}

// Debounced save on textarea input
function scheduleSave() {
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(autoSave, 1800);
  // Mark as dirty immediately
  if (!_dirty) {
    _dirty = true;
    ocrTextarea.classList.add('unsaved');
    const unsavedEl = $('unsavedIndicator');
    if (unsavedEl) unsavedEl.style.display = 'inline';
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// THUMBNAIL STRIP
// ─────────────────────────────────────────────────────────────────────────────
function buildThumbStrip() {
  if (!thumbStrip) return;
  thumbStrip.innerHTML = '';

  _images.forEach((img, idx) => {
    const div = document.createElement('div');
    div.className    = `strip-thumb ${idx === _curIndex ? 'active' : ''}`;
    div.dataset.idx  = idx;

    const imgEl = document.createElement('img');
    imgEl.src    = `/api/image/${img.filename}`;
    imgEl.alt    = img.filename;
    imgEl.loading = 'lazy';

    const dot = document.createElement('div');
    dot.className = `strip-status s-${img.ocr_status || 'pending'}`;
    dot.id        = `stripDot_${idx}`;

    div.appendChild(imgEl);
    div.appendChild(dot);
    div.addEventListener('click', () => navigateTo(idx));
    thumbStrip.appendChild(div);
  });
}

function updateStripDot(idx, status) {
  const dot = $(`stripDot_${idx}`);
  if (!dot) return;
  dot.className = `strip-status s-${status || 'pending'}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// STATUS RADIO
// ─────────────────────────────────────────────────────────────────────────────
function setStatusRadio(value) {
  statusRadios.forEach(item => {
    const radio = item.querySelector('input[type="radio"]');
    if (!radio) return;
    const isSelected = radio.value === value;
    radio.checked = isSelected;
    item.className = `status-radio-item ${isSelected ? `sel-${value}` : ''}`;
  });
}

function getStatusRadioValue() {
  for (const item of statusRadios) {
    const radio = item.querySelector('input[type="radio"]');
    if (radio && radio.checked) return radio.value;
  }
  return 'pending';
}

statusRadios.forEach(item => {
  item.addEventListener('click', () => {
    const radio = item.querySelector('input[type="radio"]');
    if (radio) {
      radio.checked = true;
      setStatusRadio(radio.value);
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// OCR STATUS BADGE
// ─────────────────────────────────────────────────────────────────────────────
function updateOcrBadge(status) {
  if (!ocrStatusBadge) return;
  const map = {
    pending:   ['badge-pending',   'Pending'],
    validated: ['badge-validated', 'Validated'],
    failed:    ['badge-failed',    'Failed'],
    skipped:   ['badge-skipped',   'Skipped'],
    running:   ['badge-running',   'OCR Running...'],
  };
  const [cls, label] = map[status] || ['badge-pending', 'Pending'];
  ocrStatusBadge.className = `badge ${cls}`;
  ocrStatusBadge.textContent = label;
}

// ─────────────────────────────────────────────────────────────────────────────
// OCR BATCH
// ─────────────────────────────────────────────────────────────────────────────
async function startOcr(retryFailed = false, singleFilename = null) {
  if (!singleFilename) {
    btnRunOcr.disabled = true;
    btnRunOcr.classList.add('is-running');
  } else {
    btnRunOcrCurr.disabled = true;
  }

  try {
    const res = await fetchJSON('/api/ocr/start', {
      method: 'POST',
      body:   JSON.stringify({
        model:        _settings.model,
        project:      _settings.project,
        concurrency:  _settings.concurrency,
        annotator:    _settings.annotator,
        retry_failed: retryFailed,
        filename:     singleFilename,
      }),
    });

    showToast(`▶ OCR started for ${res.count} images`, 'info');
    startOcrPoll(); // Start polling immediately when launched
  } catch (e) {
    showToast('Failed to start OCR: ' + e.message, 'error');
    btnRunOcr.disabled = false;
    btnRunOcr.classList.remove('is-running');
    if (btnRunOcrCurr) btnRunOcrCurr.disabled = false;
  }
}

function startOcrPoll() {
  if (_pollTimer) return; // Already polling
  _pollTimer = setInterval(pollOcrStatus, 1000);
}

function stopOcrPoll() {
  if (_pollTimer) {
    clearInterval(_pollTimer);
    _pollTimer = null;
  }
}

async function pollOcrStatus() {
  try {
    const job = await fetchJSON('/api/ocr/status');
    updateOcrProgress(job);

    if (job.status === 'completed' || job.status === 'idle') {
      btnRunOcr.disabled = false;
      btnRunOcr.classList.remove('is-running');
      if (btnRunOcrCurr) btnRunOcrCurr.disabled = false;
      stopOcrPoll(); // Stop polling when finished or idle
    } else {
      btnRunOcr.disabled = true;
      if (btnRunOcrCurr) btnRunOcrCurr.disabled = true;
      startOcrPoll(); // Make sure polling is active if backend is running
    }

    // Refresh current image OCR text when it finishes
    const img = currentImage();
    if (img) {
      const result = job.results[img.filename];

      if (result && result.status === 'running') {
        // Mark that this image is being OCR-processed right now
        _curImgOcrRunning = true;
        updateOcrBadge('running');
        setSaveState('saving', 'OCR running for this image…', '');
      }

      if (result && result.status === 'done' && _curImgOcrRunning) {
        // OCR just completed for the current image — always update textarea
        _curImgOcrRunning = false;
        ocrTextarea.value = result.text || '';
        _dirty = false;  // fresh from server, not dirty
        setSaveState('idle', 'OCR complete — review & save when ready', '');
        updateOcrBadge('pending');
        showToast('🤖 OCR done — text loaded into editor', 'success');

        // Update in-memory text
        if (_images[_curIndex]) _images[_curIndex].ocr_text = result.text || '';
      }

      if (result && result.status === 'failed' && _curImgOcrRunning) {
        _curImgOcrRunning = false;
        setSaveState('error', `OCR failed: ${result.error || 'unknown error'}`, '');
        showToast(`❌ OCR failed: ${result.error || 'unknown'}`, 'error');
      }
    }

    // Update strip dots for all running/done
    Object.entries(job.results).forEach(([filename, r]) => {
      const idx = _images.findIndex(i => i.filename === filename);
      if (idx >= 0) {
        const mappedStatus = r.status === 'done' ? 'pending' : r.status;
        updateStripDot(idx, mappedStatus);
      }
    });

  } catch (e) {
    // Silently ignore poll errors
  }
}

function updateOcrProgress(job) {
  if (cntDone)    cntDone.textContent    = job.done    || 0;
  if (cntRunning) cntRunning.textContent = job.running || 0;
  if (cntFailed)  cntFailed.textContent  = job.failed  || 0;
  if (cntQueued)  cntQueued.textContent  = job.queued  || 0;

  if (progressBar && job.total > 0) {
    const pct = Math.round(((job.done + job.failed) / job.total) * 100);
    progressBar.style.width = pct + '%';
  }

  if (btnRetry) {
    btnRetry.style.display = (job.failed > 0) ? 'inline-flex' : 'none';
  }
}

function updateStatsBar() {
  const total = _images.length;
  if (!total) return;

  // OCR Done: images where AI ran OCR (ocr_timestamp was set by AI job)
  const ocrDone  = _images.filter(i => i.ocr_timestamp).length;
  // Reviewed: images where human explicitly saved (saved_timestamp set)
  const reviewed = _images.filter(i => i.saved_timestamp).length;

  const ocrPct = Math.round((ocrDone  / total) * 100);
  const revPct = Math.round((reviewed / total) * 100);

  const ocrDoneEl  = $('statOcrDone');
  const reviewedEl = $('statReviewed');
  if (ocrDoneEl)  ocrDoneEl.textContent  = ocrDone;
  if (reviewedEl) reviewedEl.textContent = reviewed;

  const ocrBar = $('ocrProgressBarHeader');
  if (ocrBar) ocrBar.style.width = ocrPct + '%';

  const revBar = $('headerProgressBar');
  if (revBar) revBar.style.width = revPct + '%';
}

// ─────────────────────────────────────────────────────────────────────────────
// EDITED BADGE & LAST SAVED
// ─────────────────────────────────────────────────────────────────────────────
function updateEditedBadge(ocrData) {
  const badge     = $('editedBadge');
  const lastSaved = $('lastSavedLabel');

  // human_edited = server computed: text differs from original AI OCR output
  const isEdited = !!(ocrData && ocrData.human_edited);
  const savedTs  = ocrData && (ocrData.saved_timestamp || ocrData.ocr_timestamp);

  if (badge) {
    badge.style.display = isEdited ? 'inline-flex' : 'none';
  }

  if (lastSaved) {
    if (savedTs) {
      lastSaved.textContent = `Last saved: ${timeAgo(new Date(savedTs))}`;
    } else {
      lastSaved.textContent = '';
    }
  }
}

function timeAgo(date) {
  const secs = Math.floor((Date.now() - date.getTime()) / 1000);
  if (secs < 5)   return 'just now';
  if (secs < 60)  return `${secs}s ago`;
  const mins = Math.floor(secs / 60);
  if (mins < 60)  return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)   return `${hrs}h ago`;
  return date.toLocaleDateString();
}

// ─────────────────────────────────────────────────────────────────────────────
// DELETE
// ─────────────────────────────────────────────────────────────────────────────
function openDeleteModal() {
  const img = currentImage();
  if (!img) return;

  $('deleteImageName').textContent = img.filename;
  $('deleteConfirmInput').value    = '';
  $('deleteConfirmBtn').disabled   = true;

  deleteModal.classList.add('open');
  setTimeout(() => $('deleteConfirmInput').focus(), 200);
}

function closeDeleteModal() {
  deleteModal.classList.remove('open');
}

$('deleteConfirmInput')?.addEventListener('input', e => {
  $('deleteConfirmBtn').disabled = (e.target.value !== 'DELETE');
});

$('deleteConfirmBtn')?.addEventListener('click', async () => {
  const img = currentImage();
  if (!img) return;

  try {
    const res = await fetchJSON('/api/delete_image', {
      method:  'DELETE',
      body:    JSON.stringify({ filename: img.filename, confirm: 'DELETE' }),
    });

    closeDeleteModal();
    showToast(`🗑 Deleted: ${img.filename}`, 'success');

    // Remove from local list
    _images.splice(_curIndex, 1);
    buildThumbStrip();

    if (_images.length === 0) {
      mainCanvas.getContext('2d').clearRect(0, 0, mainCanvas.width, mainCanvas.height);
      ocrTextarea.value = '';
      return;
    }

    const newIdx = Math.min(_curIndex, _images.length - 1);
    await navigateTo(newIdx, false);
  } catch (e) {
    showToast('Delete failed: ' + e.message, 'error');
  }
});

$('deleteCancelBtn')?.addEventListener('click', closeDeleteModal);

// Close modal on overlay click
deleteModal?.addEventListener('click', e => {
  if (e.target === deleteModal) closeDeleteModal();
});

// ─────────────────────────────────────────────────────────────────────────────
// SETTINGS MODAL
// ─────────────────────────────────────────────────────────────────────────────
function openSettingsModal() {
  $('settingModel').value       = _settings.model       || 'gemini-2.5-flash-lite';
  $('settingProject').value     = _settings.project     || '';
  $('settingConcurrency').value = _settings.concurrency || 10;
  $('settingAnnotator').value   = _settings.annotator   || '';
  settingsModal.classList.add('open');
}

function closeSettingsModal() {
  settingsModal.classList.remove('open');
}

$('settingsSaveBtn')?.addEventListener('click', async () => {
  _settings = {
    ..._settings,
    model:        $('settingModel').value.trim(),
    project:      $('settingProject').value.trim(),
    concurrency:  parseInt($('settingConcurrency').value) || 10,
    annotator:    $('settingAnnotator').value.trim(),
  };
  await fetchJSON('/api/settings', { method: 'POST', body: JSON.stringify(_settings) });
  closeSettingsModal();
  showToast('✅ Settings saved', 'success');
});

$('settingsCancelBtn')?.addEventListener('click', closeSettingsModal);

settingsModal?.addEventListener('click', e => {
  if (e.target === settingsModal) closeSettingsModal();
});

// ─────────────────────────────────────────────────────────────────────────────
// EVENT LISTENERS
// ─────────────────────────────────────────────────────────────────────────────
function attachListeners() {
  btnPrev?.addEventListener('click',        () => navigateTo(_curIndex - 1));
  btnNext?.addEventListener('click',        () => navigateTo(_curIndex + 1));
  jumpInput?.addEventListener('keydown',    e => {
    if (e.key === 'Enter') {
      e.preventDefault();
      doJump();
    }
  });
  btnJump?.addEventListener('click',        doJump);
  btnToggleMask?.addEventListener('click',  toggleMask);
  btnRunOcr?.addEventListener('click',      () => startOcr(false));
  btnRunOcrCurr?.addEventListener('click',  () => {
    const img = currentImage();
    if (img) startOcr(false, img.filename);
  });
  btnRetry?.addEventListener('click',       () => startOcr(true));
  btnSave?.addEventListener('click',        () => autoSave().then(() => showToast('💾 Saved', 'success')));
  btnDelete?.addEventListener('click',      openDeleteModal);
  btnSettings?.addEventListener('click',   openSettingsModal);
  btnExport?.addEventListener('click',      () => window.open('/api/export', '_blank'));
  btnThemeToggle?.addEventListener('click', () => {
    const isDark = document.documentElement.classList.toggle('dark-mode');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
  });

  ocrTextarea?.addEventListener('input',   scheduleSave);
  chkContainsDiagram?.addEventListener('change', () => {
    if (diagramBadge) {
      diagramBadge.style.display = chkContainsDiagram.checked ? 'inline-flex' : 'none';
    }
    scheduleSave();
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    // Don't hijack if user is typing in an input/modal
    const tag = document.activeElement?.tagName?.toLowerCase();
    if (tag === 'input' || tag === 'select') return;
    if (deleteModal?.classList.contains('open')) return;
    if (settingsModal?.classList.contains('open')) return;

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      autoSave().then(() => {
        showToast('💾 Saved', 'success');
        navigateTo(_curIndex + 1);
      });
    }

    if (e.key === 'ArrowRight') {
      e.preventDefault();
      navigateTo(_curIndex + 1);
    }

    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      navigateTo(_curIndex - 1);
    }

    if (e.key === 'm' || e.key === 'M') {
      toggleMask();
    }
  });
}

function toggleMask() {
  _showMask = !_showMask;
  if (btnToggleMask) {
    btnToggleMask.textContent = _showMask ? '🙈 Hide Mask' : '👁 Show Mask';
  }
  const img = currentImage();
  if (img) renderMainCanvas(img);
}

// ─────────────────────────────────────────────────────────────────────────────
// UTILITIES
// ─────────────────────────────────────────────────────────────────────────────
async function fetchJSON(url, opts = {}) {
  const headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
  const res = await fetch(url, { ...opts, headers });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

let _toastTimer = null;

function showToast(msg, type = 'info') {
  let toast = $('toastEl');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toastEl';
    toast.style.cssText = `
      position: fixed; bottom: 20px; right: 20px; z-index: 9999;
      padding: 10px 16px; border-radius: 8px; font-size: 13px; font-weight: 500;
      backdrop-filter: blur(12px); transition: opacity 0.3s;
      border: 1px solid rgba(255,255,255,0.1);
      max-width: 360px; word-break: break-word;
    `;
    document.body.appendChild(toast);
  }

  const colors = {
    success: 'rgba(0,229,160,0.15)',
    error:   'rgba(255,77,109,0.15)',
    info:    'rgba(108,99,255,0.15)',
  };
  toast.style.background = colors[type] || colors.info;
  toast.style.color      = '#e8eaf0';
  toast.textContent      = msg;
  toast.style.opacity    = '1';

  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { toast.style.opacity = '0'; }, 3000);
}

// ─────────────────────────────────────────────────────────────────────────────
// BOOT
// ─────────────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', init);
