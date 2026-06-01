# AI Voice Receptionist

A complete, local, real-time voice receptionist built from scratch: it listens, transcribes,
reasons, books appointments, and talks back — and you can interrupt it mid-sentence.

A reusable voice pipeline with the receptionist built as two "slots" filled in:

- **Persona** (`receptionist.py` → `PERSONA`) — who the agent is and how it speaks.
- **Tools** (`receptionist.py` → `TOOLS` + `tools.py`) — what it can actually do.

Swap those two things and the same engine becomes any other agent.

---

## What's in the box

| File | Role |
|------|------|
| `engine.py` | The reusable voice pipeline: mic + VAD, Whisper ASR, interruptible Piper TTS, barge-in, resilience + timing. Project-agnostic. |
| `receptionist.py` | The app: persona, tool schemas, the agent loop, and `main()`. Run this. |
| `tools.py` | Receptionist capabilities: business info, availability, booking, transfer. Appointments save to `appointments.json`. |
| `requirements.txt` | Python dependencies. |

---

## Prerequisites

- Python 3.9+
- A microphone and speakers (headphones optional — echo is handled, see "Echo" below)
- An OpenAI API key
- **Apple Silicon (M1/M2/M3): everything must run as native `arm64`.** Two traps:
  1. An x86_64 Python (e.g. Homebrew under `/usr/local`) runs the ML/audio libs under Rosetta,
     which lacks AVX → `illegal hardware instruction` (SIGILL). Use an arm64 interpreter;
     `/usr/bin/python3` is a fine choice.
  2. Even with an arm64 venv, a **Terminal running under Rosetta** makes the universal `python`
     binary launch as x86_64 (it inherits the shell's arch), which then can't load the arm64
     wheels. Check your shell with `arch` — it must print `arm64`, not `i386`/`x86_64`. If it's
     wrong, run `exec arch -arm64 zsh`, or uncheck "Open using Rosetta" in the terminal app's
     Get Info. As a one-shot you can always force it: `arch -arm64 .venv/bin/python receptionist.py`.

  Confirm the running interpreter with `python -c "import platform; print(platform.machine())"` → `arm64`.

---

## Setup

1. **Install dependencies** (a virtual environment is recommended). On Apple Silicon, create the
   venv with a native arm64 interpreter:

   ```bash
   /usr/bin/python3 -m venv .venv        # arm64 on Apple Silicon
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Download a Piper voice model.** You need two files — a `.onnx` model and its
   `.onnx.json` config — in this folder. The default expected file is
   `en_US-lessac-medium.onnx`. Get voices from the Piper voices collection
   (search "piper voices huggingface"). Place both files here, or point at your own:

   ```bash
   export PIPER_MODEL=/path/to/your-voice.onnx   # optional; defaults to en_US-lessac-medium.onnx
   ```

3. **Set your API key and (optionally) model.** Copy `.env.example` to `.env` and fill it in:

   ```bash
   cp .env.example .env
   # then edit .env:
   #   OPENAI_API_KEY=sk-...
   #   OPENAI_MODEL=gpt-5.4-mini   # optional; any tool-calling chat model works
   ```

   The `.env` is loaded automatically at startup (via `python-dotenv`) — no `export` needed. You can
   still use environment variables directly if you prefer; `OPENAI_MODEL` defaults to `gpt-5.4-mini`.

---

## Run it

```bash
python receptionist.py
```

The receptionist will greet you. Then just talk:

- *"What are your hours?"* → answered from the business facts in the persona.
- *"What's open on Friday?"* → calls `check_availability`.
- *"Book me a cleaning Friday at 2 PM, my name's Sam."* → confirms, then calls `book_appointment`.
- *"I have a dental emergency."* → calls `transfer_to_human`.

Open `appointments.json` afterward to see what got booked. Try **talking over the receptionist**
while it's speaking — it should stop instantly and treat your interruption as the next turn.

Press **Ctrl+C** to quit. On the way out, the receptionist prints a per-stage latency summary (count, p50, p95
in milliseconds) for `asr` (transcription only) and `agent_turn` (the LLM), so you can see how responsive
the session was. Note these time *compute*, not the time you spend talking.

The LLM is OpenAI (default `gpt-5.4-mini`, set `OPENAI_MODEL` in `.env` to change it). OpenAI
**automatically** caches long, identical prompt prefixes, so repeated turns with the same persona
and tool schemas get a latency/cost discount with no special flags.

---

## Customizing

**Change the business:** edit `BUSINESS` and `SLOTS` in `tools.py`.

**Change the personality:** edit `PERSONA` in `receptionist.py`. Every line in it is there to
fight a specific bad default (long answers, markdown, robotic phrasing) — change the tone, keep
the spoken-style rules.

**Add a tool:** add a schema to `TOOLS`, an implementation function in `tools.py`, and one line
to `TOOLS_IMPL`. That's the entire pattern — the agent loop already handles the rest.

**Tune responsiveness:** `SILENCE_MS` (env var or `engine.py`, default 900) controls how long the
receptionist waits before deciding you're done talking — lower = snappier but cuts you off on natural
pauses; higher = more patient. If it keeps interrupting you mid-sentence, raise it (e.g. `SILENCE_MS=1200`).

**Quality vs. speed:** `ASR_MODEL` (env var or `engine.py`; `tiny.en` → `base.en` → `small.en`) trades
speed for accuracy. If transcription is wrong a lot, try `ASR_MODEL=small.en` — more accurate, but
slower on a CPU without AVX. The LLM is set by `OPENAI_MODEL` in your `.env` (defaults to `gpt-5.4-mini`, a fast,
cheap fit for short receptionist turns) — point it at a larger model if you need deeper reasoning.
(Verify current model strings in the OpenAI docs before deploying — they change with each release.)

---

## Troubleshooting

- **Can't interrupt it / it talks over you.** Barge-in uses an energy gate while the receptionist
  speaks: talk over it and it stops within ~150 ms. If it ignores you, your mic is quiet — lower
  `BARGE_RMS_MIN` (e.g. `0.008`) or `BARGE_FACTOR` in `.env`.
- **Echo / the receptionist interrupts itself.** Handled in two stages so you don't need headphones:
  at startup it plays a probe tone to **measure the speaker→mic echo coupling**, then while speaking
  it **predicts its own echo** (coupling × current playback loudness) and only treats the mic as a
  caller when the input rises `AEC_MARGIN`× above that prediction. The bar tracks how loud it is —
  high during speech, low in the gaps — so even a loud speaker can't self-trigger, while you talking
  over it still cuts in. If it still self-interrupts on very loud speakers, raise `AEC_MARGIN`; if it
  ignores you, lower it. Skip the probe tone with `CALIBRATE=0`. The strongest production fix is acoustic echo
  cancellation (AEC).
- **`Piper voice model not found`.** You haven't downloaded the `.onnx` + `.onnx.json` files, or
  the path is wrong. See Setup step 2 or set `PIPER_MODEL`.
- **Piper API errors on `synthesize`.** `piper-tts` has changed its API across versions. `engine.py`
  targets the current API and falls back to the older `synthesize_stream_raw`. If both fail, check
  your installed version with `pip show piper-tts`.
- **Silero VAD type errors.** Some versions want a torch tensor instead of a numpy array. If
  `vad(chunk)` complains, wrap the chunk: `import torch; vad(torch.from_numpy(chunk))` in `engine.py`.
- **No audio devices.** `python -c "import sounddevice; print(sounddevice.query_devices())"` lists
  your devices; make sure a default input and output exist.
- **`OPENAI_API_KEY` not set.** Put it in `.env` (Setup step 3) or export it.

---

## Where to take it next

- **Stream the LLM** for even lower latency on longer replies.
- **Real database** instead of `appointments.json` (Postgres, SQLite) for multi-user.
- **Browser or phone front-end:** stream mic audio over WebSocket to this backend; for phone,
  put it behind a telephony provider. Frameworks like Pipecat or LiveKit handle the transport
  and scaling for you.
- **Speakerphone / true echo cancellation:** the startup calibration tames mild speaker echo, but
  loud speakerphone needs real acoustic echo cancellation (AEC). On mobile this is free from the OS
  — enable iOS `AVAudioSession` `.voiceChat` mode or Android `AcousticEchoCanceler`; in the browser
  use WebRTC `getUserMedia({ audio: { echoCancellation: true } })`. On the desktop pipeline, run the
  mic through a WebRTC/SpeexDSP AEC (with the TTS output as the reference signal) before the VAD.
- **Guardrails:** confirm before any irreversible action, moderate input/output, validate tool
  arguments.
