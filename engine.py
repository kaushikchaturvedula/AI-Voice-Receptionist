"""
engine.py — the reusable voice-agent pipeline.

A project-agnostic real-time voice machine:
  always-on mic + VAD  ->  Whisper ASR  ->  [your agent]  ->
  streaming Piper TTS  ->  speaker, with barge-in interruption,
  wrapped in resilience + timing helpers.

Nothing in here knows it's a receptionist. Swap the persona + tools in
receptionist.py and this same engine powers any voice agent.
"""

import os

# torch (via silero-vad) and ctranslate2 (via faster-whisper) each bundle their own
# copy of the Intel OpenMP runtime (libiomp5). When the second one loads into the
# process, macOS OpenMP aborts with "OMP: Error #15 ... already initialized". Allow
# the duplicate so both can coexist. This MUST be set before importing torch /
# faster-whisper, so it lives at the very top of the module.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# Keep the two thread pools from oversubscribing the CPU and fighting each other.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import re
import time
import difflib
import threading
import contextlib
from collections import deque

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator

# ----------------------------------------------------------------------------
# Config — tune these to trade latency against accuracy/quality.
# ----------------------------------------------------------------------------
SR_IN = 16000                 # ASR + VAD want 16 kHz mono
CHUNK = 512                   # Silero @16kHz needs exactly 512 samples (~32 ms)
PREROLL = 10                  # chunks of audio kept before speech (~320 ms) so we don't clip
# Endpointing: trailing silence before "you're done talking". 600 ms cuts people off
# on natural mid-sentence pauses; ~900 ms is far more forgiving. Override via SILENCE_MS.
SILENCE_MS = int(os.environ.get("SILENCE_MS", "900"))
# ASR quality vs. speed: "tiny.en" < "base.en" < "small.en". On a slow CPU, base.en is the
# sweet spot; bump to small.en for noticeably better accuracy. Override via ASR_MODEL.
ASR_MODEL = os.environ.get("ASR_MODEL", "base.en")
# Playback frame size: we slice each synthesized chunk into ~46 ms writes so a barge-in
# can stop speech almost immediately instead of waiting for a whole sentence to finish.
PLAY_FRAME = 1024
# Fast barge-in energy gate. The neural VAD only flags a *fresh* speech onset, so it
# misses the most common interruption — you're already talking when the agent starts.
# While the agent speaks we instead watch raw input energy (RMS) and cut in on any
# sustained speech. The trigger is relative to the measured room noise floor (so it
# adapts to your mic) with an absolute floor as a backstop. Tune via env if the agent
# ignores you (lower these) or interrupts itself on noise (raise them).
BARGE_RMS_MIN = float(os.environ.get("BARGE_RMS_MIN", "0.015"))  # absolute RMS floor (float32 audio)
BARGE_FACTOR = float(os.environ.get("BARGE_FACTOR", "3.0"))      # ... or this multiple of room noise
BARGE_CHUNKS = int(os.environ.get("BARGE_CHUNKS", "4"))          # consecutive loud frames (~130 ms) to fire
# Startup echo calibration: play a probe tone, measure how much leaks back into the mic,
# and learn the echo coupling gain (see the AEC-style gate below). CALIBRATE=0 skips it.
CALIBRATE = os.environ.get("CALIBRATE", "1") != "0"
# Lightweight echo cancellation (double-talk detection). We know exactly what we're
# playing, so while the agent speaks the listener predicts the echo level at the mic as
# (echo_gain * current_playback_level) and only treats the input as a real interruption
# when it rises AEC_MARGIN above that prediction. The threshold therefore tracks the
# agent's own loudness — high during loud speech, low in the gaps — so even a loud
# speaker can't self-trigger, while a caller talking over it still can.
AEC_MARGIN = float(os.environ.get("AEC_MARGIN", "2.5"))
# Per-chunk decay of the predicted-echo memory after playback stops, so the gate keeps
# rejecting the ~200-300 ms acoustic echo tail before relaxing for a real caller.
ECHO_TAIL_DECAY = float(os.environ.get("ECHO_TAIL_DECAY", "0.85"))
# Verbose per-utterance diagnostics for the echo gate (temporary debugging aid).
RX_DEBUG = os.environ.get("RX_DEBUG", "0") != "0"
PIPER_MODEL = os.environ.get("PIPER_MODEL", "en_US-lessac-medium.onnx")

# ----------------------------------------------------------------------------
# Shared signals between the listener thread and the speaker.
# ----------------------------------------------------------------------------
agent_speaking = threading.Event()   # set while TTS is playing
barge_in = threading.Event()         # set when the caller interrupts
stop = threading.Event()             # set on shutdown so the listener thread exits cleanly

# Effective absolute barge-in floor (the always-on minimum sensitivity).
_barge_floor = BARGE_RMS_MIN
# Fraction of playback level that returns to the mic, measured by calibrate_echo().
# 0 means isolated (headphones) — the echo-tracking term then contributes nothing.
_echo_gain = 0.0
# Current smoothed RMS (float units) of what the speaker is playing right now; 0 when
# idle. Published by play_interruptible, read by the listener for the AEC-style gate.
_output_level = 0.0
# The last few phrases the agent spoke (newest last), for the content-based self-echo
# guard. Anything the mic transcribes that closely matches one of these is our own voice.
_recent_spoken = deque(maxlen=5)

# ----------------------------------------------------------------------------
# Lazy model loading — expensive, so we load once on first use, not at import.
# ----------------------------------------------------------------------------
_asr = None
_voice = None


def _get_asr():
    global _asr
    if _asr is None:
        log("Loading ASR model (first run downloads it)...")
        _asr = WhisperModel(ASR_MODEL, device="cpu", compute_type="int8")
    return _asr


def _get_voice():
    global _voice
    if _voice is None:
        from piper import PiperVoice
        if not os.path.exists(PIPER_MODEL):
            raise FileNotFoundError(
                f"Piper voice model not found at '{PIPER_MODEL}'. "
                "Download one (see the README) or set the PIPER_MODEL env var."
            )
        log(f"Loading Piper voice: {PIPER_MODEL}")
        _voice = PiperVoice.load(PIPER_MODEL)
    return _voice


# ----------------------------------------------------------------------------
# Observability
# ----------------------------------------------------------------------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# Per-stage latency samples (ms), accumulated across the session for the
# p50/p95 summary printed on shutdown. Plain lists -> no extra dependency.
_lat = {}


@contextlib.contextmanager
def timed(stage):
    t = time.perf_counter()
    try:
        yield
    finally:
        ms = (time.perf_counter() - t) * 1000
        _lat.setdefault(stage, []).append(ms)
        log(f"{stage}: {ms:.0f} ms")


def _percentile(samples, pct):
    """Nearest-rank percentile over a list of floats (no numpy dependency)."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[k]


def latency_summary():
    """Print count + p50 + p95 per stage. Call on shutdown to review responsiveness."""
    if not _lat:
        return
    log("--- latency summary (ms) ---")
    for stage in sorted(_lat):
        s = _lat[stage]
        log(f"{stage}: n={len(s)} p50={_percentile(s, 50):.0f} p95={_percentile(s, 95):.0f}")


# ----------------------------------------------------------------------------
# Resilience — try primary, retry, fall back, never crash on the user.
# ----------------------------------------------------------------------------
def resilient(primary, fallback, retries=1, backoff=0.3, on_total_failure=None):
    for attempt in range(retries + 1):
        try:
            return primary()
        except Exception as e:
            log(f"primary failed (attempt {attempt + 1}): {e}")
            time.sleep(backoff * (attempt + 1))
    try:
        return fallback()
    except Exception as e:
        log(f"fallback failed: {e}")
        return on_total_failure


# ----------------------------------------------------------------------------
# ASR
# ----------------------------------------------------------------------------
def transcribe(audio):
    segments, _ = _get_asr().transcribe(audio, language="en")
    return " ".join(s.text for s in segments).strip()


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).split()


def is_self_echo(text, ratio=0.8, overlap=0.8):
    """Content guard: True if `text` closely matches something the agent just said.

    Backstop to the acoustic gate — if a transcript slips through, recognize it as our
    own voice by comparing against the last few spoken phrases (substring, sequence
    similarity, or token overlap). Requires >=2 tokens so short caller turns like "yes"
    or "okay" are never dropped on coincidence.
    """
    toks = _norm(text)
    if len(toks) < 2:
        return False
    t = " ".join(toks)
    for spoken in _recent_spoken:
        s_toks = _norm(spoken)
        s = " ".join(s_toks)
        if not s:
            continue
        if t in s or s in t:
            return True
        if difflib.SequenceMatcher(None, t, s).ratio() >= ratio:
            return True
        if sum(1 for w in toks if w in s_toks) / len(toks) >= overlap:
            return True
    return False


# ----------------------------------------------------------------------------
# TTS — streaming Piper playback that can be cut off by barge-in.
# ----------------------------------------------------------------------------
def _pcm_chunks(voice, text):
    """Yield int16 PCM byte chunks. Targets current piper-tts; falls back to the older API."""
    try:
        for chunk in voice.synthesize(text):          # current piper-tts: AudioChunk objects
            yield chunk.audio_int16_bytes
        return
    except AttributeError:
        pass
    for raw in voice.synthesize_stream_raw(text):     # older piper-tts: raw int16 bytes
        yield raw


def play_interruptible(text):
    """Speak `text`, but stop instantly if the caller barges in."""
    global _output_level
    if not text:
        return
    voice = _get_voice()
    sr = voice.config.sample_rate
    _recent_spoken.append(text)          # remember it so we can recognize its echo
    agent_speaking.set()
    barge_in.clear()
    try:
        with sd.OutputStream(samplerate=sr, channels=1, dtype="int16") as out:
            for pcm in _pcm_chunks(voice, text):
                # Piper hands us a whole sentence as ONE chunk, so we can't just check
                # barge_in between chunks — by then the sentence has already played.
                # Slice into small frames and check before each short write, so an
                # interruption stops playback within ~one frame (~46 ms).
                samples = np.frombuffer(pcm, dtype=np.int16)
                for i in range(0, len(samples), PLAY_FRAME):
                    if barge_in.is_set():
                        return                         # <-- the interruption
                    frame = samples[i:i + PLAY_FRAME]
                    # Publish this frame's level (float units) for the listener's
                    # echo-tracking gate. Peak-hold with decay so the estimate lingers
                    # ~150 ms, covering the acoustic + buffer delay of the echo tail.
                    lvl = _rms(frame.astype(np.float32) / 32768.0)
                    _output_level = max(lvl, _output_level * 0.6)
                    out.write(frame)
    finally:
        _output_level = 0.0
        agent_speaking.clear()


# ----------------------------------------------------------------------------
# Echo calibration — detect speakers vs. headphones by measurement, not guesswork.
# ----------------------------------------------------------------------------
def _rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def calibrate_echo(probe_sec=0.5, couple_ratio=3.0, margin=1.6):
    """Measure speaker->mic echo and auto-tune the barge-in floor.

    Plays a short swept tone while recording the mic. If the mic hears it far louder
    than the room's ambient noise, the output is acoustically coupled to the input
    (open speakers / speakerphone), so we raise the barge-in floor above the echo level
    — that way the agent's own voice can't self-trigger an interruption, while a caller
    speaking *over* the echo still can. If the probe stays quiet (headphones, headset,
    or a phone held to the ear) we keep the sensitive default. Device-name sniffing is
    avoided on purpose: it's unreliable on desktop and meaningless on mobile, whereas
    this physical measurement works anywhere there's a mic and a speaker.

    Returns a profile dict and updates the module-level barge floor as a side effect.
    """
    global _echo_gain
    if not CALIBRATE:
        return {"mode": "disabled", "echo_gain": _echo_gain}
    try:
        log("Calibrating audio (you'll hear a short test tone)...")
        ambient = sd.rec(int(0.4 * SR_IN), samplerate=SR_IN, channels=1, dtype="float32")
        sd.wait()
        noise = _rms(ambient.flatten())
        t = np.linspace(0, probe_sec, int(probe_sec * SR_IN), endpoint=False, dtype=np.float32)
        probe = (0.35 * np.sin(2 * np.pi * (400 + 1600 * t / probe_sec) * t)).astype(np.float32)
        probe_rms = _rms(probe)
        rec = sd.playrec(probe, samplerate=SR_IN, channels=1, dtype="float32")
        sd.wait()
        echo = _rms(rec.flatten())
    except Exception as e:
        log(f"echo calibration skipped ({e}); using default barge-in sensitivity.")
        return {"mode": "unknown", "echo_gain": _echo_gain}

    coupling = echo / max(noise, 1e-5)
    if coupling >= couple_ratio:
        # Open speakers: learn the coupling gain so the AEC-style gate can predict and
        # subtract the agent's own echo at runtime (scales with how loud it's speaking).
        _echo_gain = echo / max(probe_rms, 1e-6)
        mode = "speakers (coupled)"
    else:
        _echo_gain = 0.0                          # isolated: no echo to model
        mode = "headphones/headset (isolated)"
    log(f"Audio setup: {mode} — noise={noise:.4f}, echo={echo:.4f} ({coupling:.1f}x), "
        f"echo_gain={_echo_gain:.3f}")
    return {"mode": mode, "noise": noise, "echo": echo, "coupling": coupling,
            "echo_gain": _echo_gain}


# ----------------------------------------------------------------------------
# Always-on listener — runs in its own thread.
# ----------------------------------------------------------------------------
def listener(on_utterance):
    """Stream the mic forever. Emit complete utterances; interrupt the agent on barge-in."""
    vad = VADIterator(
        load_silero_vad(),
        sampling_rate=SR_IN,
        min_silence_duration_ms=SILENCE_MS,
        speech_pad_ms=150,
    )
    buf, preroll, capturing = [], [], False
    noise_floor = BARGE_RMS_MIN                        # adapts to the room during quiet moments
    loud = 0                                           # consecutive loud frames while the agent speaks
    echo_mem = 0.0                                     # decaying estimate of echo at the mic
    cap_tts = cap_excess = False                       # per-utterance: overlapped TTS? exceeded echo?
    with sd.InputStream(samplerate=SR_IN, channels=1, dtype="float32", blocksize=CHUNK) as stream:
        while not stop.is_set():
            chunk = stream.read(CHUNK)[0].flatten()
            rms = float(np.sqrt(np.mean(chunk * chunk)))

            # Predicted echo of our own playback at the mic. Peak-hold + decay so it
            # also covers the acoustic echo TAIL for a few hundred ms after TTS stops
            # (when _output_level has already dropped to 0). Zero on headphones.
            echo_mem = max(_echo_gain * _output_level, echo_mem * ECHO_TAIL_DECAY)
            # A real caller must clear this bar; the agent's own echo sits below it.
            threshold = max(_barge_floor, noise_floor * BARGE_FACTOR, echo_mem * AEC_MARGIN)
            tts_active = agent_speaking.is_set() or echo_mem > _barge_floor

            # --- Barge-in: interrupt playback when a caller speaks over it -----------
            if agent_speaking.is_set():
                if rms > threshold:
                    loud += 1
                    if loud >= BARGE_CHUNKS:
                        barge_in.set()                 # caller cut in -> stop the agent
                else:
                    loud = 0
            else:
                loud = 0
                if not capturing and not tts_active:   # learn the room's quiet level
                    noise_floor = 0.95 * noise_floor + 0.05 * rms

            event = vad(chunk)
            if event and "start" in event:
                # Do NOT barge in here. On coupled speakers the agent's own echo also
                # produces a VAD "start", and setting barge_in on it would self-interrupt
                # the agent (it was cutting the greeting off after one word). The energy
                # gate above is the sole barge-in trigger: it fires only when input rises
                # above the predicted echo, so a caller interrupts but echo never does.
                capturing, buf = True, preroll[:]      # keep pre-roll so we don't clip the first word
                cap_tts = cap_excess = False           # reset per-utterance echo trackers
            if capturing:
                buf.append(chunk)
                if tts_active:
                    cap_tts = True                     # this utterance overlapped our playback
                if rms > threshold:
                    cap_excess = True                  # ...and rose above the predicted echo
            else:
                preroll.append(chunk)
                preroll[:] = preroll[-PREROLL:]
            if event and "end" in event:
                vad.reset_states()
                capturing = False
                if buf:
                    utt = np.concatenate(buf)
                    # Self-echo iff it overlapped our playback but never rose above the
                    # predicted echo. A caller talking OVER the agent clears the bar
                    # (cap_excess) and is kept — that's a genuine barge-in, not echo.
                    is_echo = cap_tts and not cap_excess
                    if RX_DEBUG:
                        log(f"[capture] tts_overlap={cap_tts} excess={cap_excess} "
                            f"utt_rms={_rms(utt):.4f} echo~{echo_mem:.4f} thr={threshold:.4f} "
                            f"-> {'DISCARD (self-echo)' if is_echo else 'ACCEPT'}")
                    if not is_echo:
                        on_utterance(utt)
                buf = []
