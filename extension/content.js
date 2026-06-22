(function () {
  // Clean up previous injection if any (content script persists across
  // extension reloads, so background.js sends subtitle-stop before inject)
  const alreadyInjected = document.getElementById('vocaltranslator-overlay');

  let ws = null;
  let overlay = null;
  let statusEl = null;

  // ── One-time DOM setup ──────────────────────────────────
  function setupDOM() {
    if (overlay) return;
    overlay = document.createElement('div');
    overlay.id = 'vocaltranslator-overlay';

    const closeBtn = document.createElement('button');
    closeBtn.className = 'vt-close';
    closeBtn.textContent = '×';
    closeBtn.title = 'Hide subtitles';
    overlay.appendChild(closeBtn);

    const dragHandle = document.createElement('div');
    dragHandle.className = 'vt-drag-handle';
    overlay.appendChild(dragHandle);

    statusEl = document.createElement('div');
    statusEl.className = 'vt-status';
    statusEl.style.display = 'none';
    overlay.appendChild(statusEl);

    document.body.appendChild(overlay);

    // Drag
    let dragging = false, dragStartY = 0, overlayStartBottom = 80;
    dragHandle.addEventListener('mousedown', (e) => {
      dragging = true;
      dragStartY = e.clientY;
      overlayStartBottom = parseInt(overlay.style.bottom) || 80;
      e.preventDefault();
    });
    document.addEventListener('mousemove', (e) => {
      if (!dragging) return;
      overlay.style.bottom = Math.max(0, overlayStartBottom + dragStartY - e.clientY) + 'px';
    });
    document.addEventListener('mouseup', () => { dragging = false; });

    closeBtn.addEventListener('click', () => {
      overlay.style.display = overlay.style.display === 'none' ? '' : 'none';
    });
  }

  if (!alreadyInjected) {
    setupDOM();
    console.log('[VT] DOM setup done, overlay:', !!getOverlay());
  } else {
    console.log('[VT] DOM already exists');
  }

  // ── Shared helpers ─────────────────────────────────────
  function getOverlay() {
    if (!overlay) overlay = document.getElementById('vocaltranslator-overlay');
    return overlay;
  }
  function getStatusEl() {
    if (!statusEl) statusEl = getOverlay() && getOverlay().querySelector('.vt-status');
    return statusEl;
  }

  const MAX_VISIBLE = 3;
  const FADE_MS = 400;

  // ── Streaming subtitle state ──────────────────────────────
  const _pending = {};  // { id: { el, text } }

  function _fadeOut(el) {
    if (!el || !el.parentNode) return;
    el.classList.add('fading');
    setTimeout(() => { if (el.parentNode) el.remove(); }, FADE_MS);
  }

  function _trimSubtitles(ov) {
    // Remove oldest DONE subtitles if over MAX_VISIBLE
    const all = ov.querySelectorAll('.vt-subtitle');
    const done = Array.from(all).filter(
      el => el.classList.contains('done') && !el.classList.contains('fading')
    );
    while (done.length > MAX_VISIBLE) {
      _fadeOut(done.shift());
    }
  }

  function showSubtitle(text, id) {
    const ov = getOverlay();
    const st = getStatusEl();
    if (!ov || !st) {
      console.log('[VT] showSubtitle failed: ov=', !!ov, 'st=', !!st);
      return null;
    }
    const el = document.createElement('div');
    el.className = 'vt-subtitle';
    el.textContent = text;
    ov.insertBefore(el, st);
    console.log('[VT] element created:', text.slice(0, 30), 'id=', id || 'none', 'visible=', !!el.parentNode);
    if (id) {
      if (_pending[id]) clearTimeout(_pending[id]._timer);
      _pending[id] = { el, text, _timer: null };
    }
    return el;
  }

  function showStatus(text) {
    const st = getStatusEl();
    if (!st) return;
    st.style.display = text ? 'block' : 'none';
    st.textContent = text;
    if (text) setTimeout(() => { st.style.display = 'none'; }, 3000);
  }

  // ── WebSocket client (receive only) ────────────────────
  function connectWebSocket(serverPort) {
    if (ws) {
      ws.close();
      ws = null;
    }

    ws = new WebSocket(`ws://127.0.0.1:${serverPort}/ws`);

    console.log('[VT] Connecting to ws://127.0.0.1:' + serverPort + '/ws');
    ws.onopen = () => { console.log('[VT] WS connected'); showStatus('Connected'); };
    ws.onclose = () => { console.log('[VT] WS disconnected'); showStatus('Disconnected'); };
    ws.onerror = (e) => { console.log('[VT] WS error'); showStatus('Connection error'); };
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        console.log('[VT] WS:', msg.type, msg.id || '', msg.text ? msg.text.slice(0, 40) : '');
        switch (msg.type) {
          case 'sentence': {
            const el = showSubtitle(msg.text);
            if (el) {
              el.classList.add('done');
              _trimSubtitles(getOverlay());
            }
            break;
          }
          case 'sentence_start': {
            console.log('[VT] start:', msg.id);
            const ov = getOverlay();
            _trimSubtitles(ov);
            const el = showSubtitle('', msg.id);
            if (el) el.classList.add('streaming');
            break;
          }
          case 'token': {
            const p = _pending[msg.id];
            if (p) {
              p.text += msg.text;
              p.el.textContent = p.text;
            } else {
              console.log('[VT] token for unknown id:', msg.id);
            }
            break;
          }
          case 'sentence_end': {
            const p = _pending[msg.id];
            if (p) {
              if (p._timer) clearTimeout(p._timer);
              p.el.textContent = msg.text || p.text;
              p.el.classList.remove('streaming');
              p.el.classList.add('done');
              delete _pending[msg.id];
            } else {
              console.log('[VT] sentence_end for unknown id:', msg.id);
            }
            break;
          }
          case 'status':
            if (msg.status === 'listening') showStatus('Listening...');
            break;
          case 'error':
            console.log('[VT] error:', msg.message);
            showStatus(msg.message);
            break;
        }
      } catch (e) { console.log('[VT] parse error:', e); }
    };
  }

  function disconnectWebSocket() {
    if (ws) { ws.close(); ws = null; }
    showStatus('Stopped');
  }

  // ── Message listener ───────────────────────────────────
  let _listenerRegistered = false;
  if (!_listenerRegistered) {
    _listenerRegistered = true;
    chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
      if (msg.type === 'ping') {
        sendResponse({ ok: true });
        return true;
      }
      if (msg.type === 'subtitle-start') {
        connectWebSocket(msg.port || 8765);
        sendResponse({ ok: true });
        return true;
      }
      if (msg.type === 'subtitle-stop') {
        disconnectWebSocket();
        sendResponse({ ok: true });
        return true;
      }
      return false;
    });
  }
})();
