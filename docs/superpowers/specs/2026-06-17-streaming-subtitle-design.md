# Streaming Subtitle Design

**Date**: 2026-06-17
**Status**: Draft

## Goal

Replace per-segment batch translation with streaming token-by-token translation
for a smoother subtitle experience. Viewers see subtitles appear gradually
(typewriter effect) instead of waiting for entire speech segments to complete.

## Current State

- VAD segments speech based on silence gaps (1500ms default)
- Whisper transcribes entire segment → LLM translates all at once → single WebSocket `sentence` event → subtitle appears as block
- Latency: viewer waits for segment end + Whisper + LLM round-trip before seeing anything
- Long utterances (up to 30s) cause long blank periods followed by text dump

## Target Behavior

- VAD flushes faster (500ms silence)
- Whisper output split into individual sentences
- Each sentence streamed token-by-token via LLM streaming API
- Client renders typewriter effect: italic while streaming, bold when complete
- Previous 2-3 translated sentences sent as context to improve translation quality

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Stream granularity | Token-level (typewriter) | Smoothest UX, lowest perceived latency |
| Sentence splitting | Post-Whisper regex split | Smaller chunks = faster first display |
| Context window | 3 previous translations | Balance quality vs API cost |
| In-progress vs done | Italic/dim vs bold | User can distinguish live vs finalized text |
| Architecture | Pipeline generator stream end-to-end | Single source of truth, no duplicate LLM calls |
| VAD silence | 500ms (was 1500ms) | More frequent segments + sentence splitting handles the rest |
| Max speech duration | 10s (was 30s) | Prevents very long segments even with no silence |

## New WebSocket Event Types

```json
// Cũ (giữ nguyên)
{"type": "status", "status": "listening"}
{"type": "error", "message": "..."}

// Mới — streaming lifecycle
{"type": "sentence_start", "id": "a1b2c3d4"}
{"type": "token",          "id": "a1b2c3d4", "text": "Xin"}
{"type": "sentence_end",   "id": "a1b2c3d4", "text": "Xin chào bạn"}
```

`sentence_end` includes the full translated text so the client can
replace any incremental build artifacts with the finalized string.

## Data Flow (New)

```
Audio chunk (4096 samples, 256ms)
  → VAD.process() — 512-sample frames, 500ms silence flush, 10s max
  → Completed segment (sr, np.ndarray)
  → WhisperSTT.transcribe() — faster-whisper, CUDA, asyncio.to_thread
  → English text: "Hello. How are you?"
  → _split_sentences() → ["Hello.", "How are you?"]
  → For each sentence:
      1. yield {"type": "sentence_start", "id": "..."}
      2. translator.translate_stream(sentence, context=last_3)
         → async for token: yield {"type": "token", ...}
      3. yield {"type": "sentence_end", "id": "...", "text": full}
  → broadcast_event() sends each event to all WebSocket clients
  → content.js state machine renders streaming/done subtitles
```

## Code Changes

### 1. `server/config.py`
- VAD default: `min_silence_duration_ms` 1500 → 500
- VAD default: `max_speech_duration_s` 30.0 → 10.0

### 2. `server/pipeline.py`

#### New: `_split_sentences(text: str) -> list[str]`
Simple regex split on `. ! ?` followed by whitespace.

#### Modified: `OpenAICompatibleTranslator`
- Add `self._context: deque[str]` with maxlen=3
- Add `async translate_stream(text, source, target)` — async generator yielding tokens via `stream=True`
- Context injected into system prompt when available
- After stream completes, full translation appended to context buffer
- Existing `translate()` preserved for backward compatibility

#### Modified: `Pipeline.process_chunk()`
- After Whisper transcription, split text into sentences
- For each sentence, yield `sentence_start`, stream tokens, yield `sentence_end`
- Sentence IDs generated via `uuid.uuid4().hex[:8]`

### 3. `server/server.py`
- Update logging in `audio_processing_loop()` to handle new event types
- No structural changes needed — already forwards all events from generator

### 4. `extension/content.js`
- Add `_pending` map: `{id: {el, text}}`
- `sentence_start`: create empty subtitle element, add `.streaming` class, store in `_pending`
- `token`: append text to element, update DOM
- `sentence_end`: replace with full text, swap `.streaming` → `.done`, remove from `_pending`

### 5. `extension/overlay.css`
- `.vt-subtitle.streaming`: italic, opacity 0.75
- `.vt-subtitle.done`: font-weight 600, opacity 1.0, transition 0.2s

## Edge Cases

- **Empty transcription**: Whisper returns empty string → skip, no events emitted
- **API connection drop**: `translate_stream` raises → yield `{"type": "error", ...}`, context buffer NOT updated (partial translation discarded)
- **Client disconnect mid-stream**: Server continues streaming; stale `_pending` entries on reconnect handled by sentence_end never arriving — new `sentence_start` with different IDs won't conflict
- **Rapid speech with no silence**: Max 10s segment limit prevents buffer bloat; sentence splitter still breaks into displayable chunks
- **Token with empty content**: LLM sometimes yields `delta.content = ""` → skip, don't send to client
- **Single-word sentences**: Sentence splitter handles "Yes." and "No." correctly
- **Non-English punctuation**: Splitter is English-specific; other languages handled by Whisper's punctuation output

## Rollback Plan

If streaming causes issues:
1. Revert `config.py` VAD defaults (1500ms, 30s)
2. Revert `pipeline.py` to use `translate()` instead of `translate_stream()`
3. Revert `content.js` to only handle `sentence` event type
4. No database migration needed — config changes are runtime defaults only

## Out of Scope

- Word-level timestamps from Whisper (not needed with token streaming)
- TTS / audio output
- Multi-language support beyond en→vi
- Client-side buffering/smoothing (server-driven streaming is sufficient)
