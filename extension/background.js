// State
let activeTabId = null;
let port = 8765;
let isCapturing = false;

// Persist capture state so popup shows correct status on reopen
function setCaptureState(active) {
  isCapturing = active;
  chrome.storage.local.set({ isCapturing: active, captureTabId: activeTabId });
}

// Handle messages from popup
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === 'checkServer') {
    checkServerConnection(msg.port || 8765)
      .then(ok => sendResponse({ ok }))
      .catch(() => sendResponse({ ok: false }));
    return true; // async response
  }
  if (msg.action === 'getState') {
    sendResponse({ isCapturing, activeTabId, port });
    return true;
  }
  if (msg.action === 'start') {
    port = msg.port || 8765;
    startSubtitleOverlay(msg.tabId);
    sendResponse({ ok: true });
  } else if (msg.action === 'stop') {
    stopSubtitleOverlay();
    sendResponse({ ok: true });
  }
  return true;
});

async function checkServerConnection(serverPort) {
  try {
    const resp = await fetch(`http://127.0.0.1:${serverPort}/admin`, {
      signal: AbortSignal.timeout(3000),
    });
    return resp.ok;
  } catch {
    return false;
  }
}

async function startSubtitleOverlay(tabId) {
  // Always stop previous content script first — content scripts persist
  // across extension reloads, so old code may still be running.
  await chrome.tabs.sendMessage(tabId, { type: 'subtitle-stop' }).catch(() => {});
  await new Promise(r => setTimeout(r, 200));

  activeTabId = tabId;

  // Inject content script while popup is still open (tab accessible)
  try {
    await injectContentScript(tabId);
  } catch (err) {
    console.error('startSubtitleOverlay: inject failed:', err.message);
    setCaptureState(false);
    updatePopupStatus('stopped');
    return;
  }

  // Retry sendMessage up to 3 times with 200ms delay (Teams SPA may
  // briefly lose content script context during navigation/focus)
  for (let attempt = 1; attempt <= 3; attempt++) {
    try {
      await chrome.tabs.sendMessage(tabId, {
        type: 'subtitle-start',
        port: port,
      });
      console.log('startSubtitleOverlay: sendMessage succeeded on attempt', attempt);
      break;
    } catch (sendErr) {
      console.log('startSubtitleOverlay: sendMessage attempt', attempt, 'failed:', sendErr.message);
      if (attempt === 3) throw sendErr;
      await new Promise(r => setTimeout(r, 200));
    }
  }

  setCaptureState(true);
  updatePopupStatus('capturing');
}

async function injectContentScript(tabId) {
  // Always inject — content scripts persist across extension reloads,
  // so a "ping success" would mean stale code is still running.
  try {
    const result = await chrome.scripting.executeScript({
      target: { tabId: tabId },
      files: ['content.js'],
    });
    console.log('injectContentScript: executeScript result:', JSON.stringify(result));
  } catch (err) {
    console.error('injectContentScript: executeScript FAILED:', err.message);
    throw err;
  }
  try {
    await chrome.scripting.insertCSS({
      target: { tabId: tabId },
      files: ['overlay.css'],
    });
    console.log('injectContentScript: insertCSS ok');
  } catch (err) {
    console.error('injectContentScript: insertCSS FAILED:', err.message);
    throw err;
  }
}

function stopSubtitleOverlay() {
  setCaptureState(false);
  if (activeTabId) {
    chrome.tabs.sendMessage(activeTabId, { type: 'subtitle-stop' }).catch(() => {});
    activeTabId = null;
  }
  updatePopupStatus('stopped');
}

function updatePopupStatus(status) {
  chrome.runtime.sendMessage({
    type: 'status',
    status: status,
  }).catch(() => {}); // Popup may not be open
}

// Clean up when Teams tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  if (tabId === activeTabId) {
    stopSubtitleOverlay();
  }
});
