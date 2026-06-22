const PORT = document.getElementById('server-port');
const STATUS_DOT = document.querySelector('.dot');
const STATUS_TEXT = document.getElementById('server-text');
const TOGGLE_BTN = document.getElementById('toggle-btn');
const SOURCE_LANG = document.getElementById('source-lang');
const TARGET_LANG = document.getElementById('target-lang');

let isRunning = false;

// On popup open: load settings + check server + check if already capturing
chrome.storage.local.get(
  ['port', 'sourceLang', 'targetLang', 'isCapturing'],
  (items) => {
    if (items.port) PORT.value = items.port;
    if (items.sourceLang) SOURCE_LANG.value = items.sourceLang;
    if (items.targetLang) TARGET_LANG.value = items.targetLang;

    // Restore capture state if background is already capturing
    if (items.isCapturing) {
      isRunning = true;
      TOGGLE_BTN.textContent = 'Stop';
      TOGGLE_BTN.className = 'btn-stop';
      TOGGLE_BTN.disabled = false;
      STATUS_DOT.className = 'dot connected';
      STATUS_TEXT.textContent = 'Capturing...';
    } else {
      checkServer();
    }
  }
);

PORT.addEventListener('change', () => {
  chrome.storage.local.set({ port: PORT.value });
  if (!isRunning) checkServer();
});

SOURCE_LANG.addEventListener('change', () => {
  chrome.storage.local.set({ sourceLang: SOURCE_LANG.value });
});

TARGET_LANG.addEventListener('change', () => {
  chrome.storage.local.set({ targetLang: TARGET_LANG.value });
});

function checkServer() {
  STATUS_DOT.className = 'dot checking';
  STATUS_TEXT.textContent = 'Checking...';
  TOGGLE_BTN.disabled = true;

  chrome.runtime.sendMessage(
    { action: 'checkServer', port: parseInt(PORT.value) },
    (response) => {
      if (chrome.runtime.lastError || !response || !response.ok) {
        STATUS_DOT.className = 'dot disconnected';
        STATUS_TEXT.textContent = 'Server not running';
        TOGGLE_BTN.disabled = true;
        return;
      }
      STATUS_DOT.className = 'dot connected';
      STATUS_TEXT.textContent = 'Connected';
      TOGGLE_BTN.disabled = false;
    }
  );
}

TOGGLE_BTN.addEventListener('click', async () => {
  if (isRunning) {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.runtime.sendMessage({ action: 'stop', tabId: tab.id });
    isRunning = false;
    TOGGLE_BTN.textContent = 'Start';
    TOGGLE_BTN.className = 'btn-start';
    STATUS_TEXT.textContent = 'Connected';
    return;
  }

  // Start capture
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (
    !tab.url.includes('teams.microsoft.com') &&
    !tab.url.includes('teams.live.com')
  ) {
    STATUS_TEXT.textContent = 'Open Teams meeting first';
    return;
  }

  chrome.runtime.sendMessage({
    action: 'start',
    tabId: tab.id,
    port: parseInt(PORT.value),
    sourceLang: SOURCE_LANG.value,
    targetLang: TARGET_LANG.value,
  });

  // Popup will close when background focuses the Teams tab.
  // State is persisted in chrome.storage, restored on next open.
  isRunning = true;
  TOGGLE_BTN.textContent = 'Stop';
  TOGGLE_BTN.className = 'btn-stop';
  STATUS_TEXT.textContent = 'Starting...';
});

// Listen for status updates from background
chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === 'status') {
    if (msg.status === 'capturing') {
      isRunning = true;
      TOGGLE_BTN.textContent = 'Stop';
      TOGGLE_BTN.className = 'btn-stop';
      STATUS_TEXT.textContent = 'Capturing...';
    } else if (msg.status === 'stopped') {
      isRunning = false;
      TOGGLE_BTN.textContent = 'Start';
      TOGGLE_BTN.className = 'btn-start';
      if (STATUS_TEXT.textContent !== 'Server not running') {
        STATUS_TEXT.textContent = 'Connected';
      }
    }
  }
});
