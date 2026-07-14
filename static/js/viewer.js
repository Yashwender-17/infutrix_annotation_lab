/**
 * viewer.js — Canvas-based YOLO polygon overlay renderer
 * Used by both the Explorer grid thumbnails and the Dashboard main view.
 */

'use strict';

/**
 * Draw an image with semi-transparent polygon overlays on a canvas.
 *
 * @param {HTMLCanvasElement} canvas  - Target canvas element
 * @param {string}            imgSrc  - URL to the image (via /api/image/<filename>)
 * @param {Array}             polygons - Array of polygons, each = [[x_norm, y_norm], ...]
 * @param {Object}            opts
 *   @param {boolean} opts.showMask  - Whether to draw the overlay (default true)
 *   @param {string}  opts.fillColor - RGBA fill (default 'rgba(108,99,255,0.30)')
 *   @param {string}  opts.stroke    - Stroke color (default 'rgba(0,229,160,0.85)')
 *   @param {number}  opts.lineWidth - Stroke width in px (default 2)
 * @returns {Promise<void>}
 */
export async function drawAnnotatedImage(canvas, imgSrc, polygons, opts = {}) {
  const {
    showMask  = true,
    fillColor = 'rgba(108,99,255,0.30)',
    stroke    = 'rgba(0,229,160,0.85)',
    lineWidth = 2,
  } = opts;

  return new Promise((resolve, reject) => {
    const img = new Image();
    // No crossOrigin needed — same-origin Flask server

    img.onload = () => {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;

      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw base image
      ctx.drawImage(img, 0, 0);

      if (showMask && polygons && polygons.length > 0) {
        const w = canvas.width;
        const h = canvas.height;

        for (const poly of polygons) {
          if (!poly || poly.length < 3) continue;

          ctx.beginPath();
          const [x0, y0] = poly[0];
          ctx.moveTo(x0 * w, y0 * h);

          for (let i = 1; i < poly.length; i++) {
            const [xi, yi] = poly[i];
            ctx.lineTo(xi * w, yi * h);
          }
          ctx.closePath();

          ctx.fillStyle   = fillColor;
          ctx.fill();

          ctx.strokeStyle = stroke;
          ctx.lineWidth   = lineWidth;
          ctx.stroke();
        }
      }

      resolve();
    };

    img.onerror = () => reject(new Error(`Failed to load image: ${imgSrc}`));
    img.src     = imgSrc;
  });
}


/**
 * Render all thumbnail cards in the image grid.
 * Each card has a <canvas> that displays the image + overlay.
 *
 * @param {Array}    images   - Array of image objects from /api/images
 * @param {string}   containerId - ID of the grid container element
 * @param {Function} onSelect - Callback(image) when a thumbnail is clicked
 */
export function renderGrid(images, containerId, onSelect) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = '';

  for (const img of images) {
    const card = document.createElement('div');
    card.className = 'thumb-card fade-in';
    card.dataset.filename = img.filename;

    const statusClass = {
      validated: 'badge-validated',
      pending:   'badge-pending',
      failed:    'badge-failed',
      skipped:   'badge-skipped',
    }[img.ocr_status || 'pending'] || 'badge-pending';

    const diagramBadge = img.contains_diagram
      ? `<span class="badge" style="background:rgba(37,99,235,0.15);color:var(--blue);border:1px solid rgba(37,99,235,0.25);">📊 Diagram</span>`
      : '';

    card.innerHTML = `
      <div class="thumb-canvas-wrap">
        <canvas id="canvas_${img.id}" width="320" height="240"></canvas>
      </div>
      <div class="thumb-info">
        <div class="thumb-filename mono">${img.filename}</div>
        <div class="thumb-meta" style="display:flex;align-items:center;gap:4px;">
          <span class="badge ${statusClass}">${img.ocr_status || 'pending'}</span>
          ${diagramBadge}
          <span style="flex:1"></span>
          <span class="mono" style="font-size:10px;color:var(--text-muted)">${img.annotation_count}↗</span>
        </div>
      </div>
    `;

    card.addEventListener('click', () => onSelect(img));
    container.appendChild(card);

    // Lazy-load canvas render
    _lazyRenderThumb(img);
  }
}


/**
 * Lazily load and render a single thumbnail canvas using IntersectionObserver.
 */
const _renderedThumbs = new Set();

function _lazyRenderThumb(img) {
  const canvas = document.getElementById(`canvas_${img.id}`);
  if (!canvas) return;

  const observer = new IntersectionObserver(async (entries, obs) => {
    for (const entry of entries) {
      if (!entry.isIntersecting) continue;
      obs.unobserve(entry.target);

      if (_renderedThumbs.has(img.filename)) continue;
      _renderedThumbs.add(img.filename);

      try {
        const polygons = await _fetchPolygons(img.filename);
        await drawAnnotatedImage(canvas, `/api/image/${img.filename}`, polygons, {
          showMask: true,
          lineWidth: 1.5,
        });
      } catch (e) {
        // Fallback: just draw a placeholder
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#111';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#333';
        ctx.font = '12px monospace';
        ctx.fillText('⚠', 10, 20);
      }
    }
  }, { rootMargin: '100px' });

  observer.observe(canvas);
}


/**
 * Cache of polygon data keyed by filename.
 */
const _polygonCache = {};

async function _fetchPolygons(filename) {
  if (_polygonCache[filename] !== undefined) return _polygonCache[filename];
  try {
    const resp = await fetch(`/api/annotations/${filename}`);
    const data = await resp.json();
    _polygonCache[filename] = data;
    return data;
  } catch {
    _polygonCache[filename] = [];
    return [];
  }
}


/**
 * Pre-fetch polygons for a list of images (warm up cache).
 */
export async function prefetchPolygons(images) {
  await Promise.all(images.map(img => _fetchPolygons(img.filename)));
}


/**
 * Clear the lazy-render cache (call when filters change and grid is rebuilt).
 */
export function clearRenderCache() {
  _renderedThumbs.clear();
}


/**
 * Fetch polygons for a filename (public, cached).
 */
export { _fetchPolygons as fetchPolygons };
