"""
receptionist.py — the AI receptionist application.

This file IS the two slots that turn the generic engine into a receptionist:
  1) PERSONA  — who the agent is and how it talks (the system prompt)
  2) TOOLS    — what it can actually do (schemas the model sees + impls it calls)

Everything else is the reusable engine. To build a different voice agent, copy
this file, rewrite PERSONA and TOOLS, and point it at the same engine.
"""

import os
import json
import queue
import datetime
import threading

from dotenv import load_dotenv
from openai import OpenAI

import engine
import tools

load_dotenv()                                  # pull OPENAI_API_KEY / OPENAI_MODEL from a local .env
client = OpenAI()                              # reads OPENAI_API_KEY from the environment

# Default model; override per deployment with OPENAI_MODEL in your .env (any chat model works).
PRIMARY_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
MAX_TURNS = 16                                  # sliding-window memory cap (in messages*2)

# ---------------------------------------------------------------------------
# SLOT 1: PERSONA
# ---------------------------------------------------------------------------
PERSONA = f"""You are the friendly AI Voice Receptionist for {tools.BUSINESS['name']}.

You are speaking to callers OUT LOUD; your words are read aloud by a text-to-speech
engine. Therefore:
- Reply in one or two short, natural spoken sentences.
- Never use markdown, lists, bullet points, headers, symbols, or emoji.
- Spell out numbers and times the way a person says them.
- Use contractions and a warm, professional tone.

About the business:
- Hours: {tools.BUSINESS['hours']}
- Address: {tools.BUSINESS['address']}
- Phone: {tools.BUSINESS['phone']}
- Services: {tools.BUSINESS['services']}

Answer common questions (hours, location, services) directly from the facts above.
To check open times or to book an appointment, use your tools. Always confirm the
caller's name, the date, and the time back to them before booking. Convert relative
dates like "tomorrow" or "Friday" into a YYYY-MM-DD date using today's date, which is
given to you each turn. If a caller is upset, has an emergency, or asks for something
you cannot handle, use the transfer tool. Never leave a caller sitting in silence."""

# ---------------------------------------------------------------------------
# SLOT 2: TOOLS  (schemas the model sees)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Look up the open appointment times for a specific day. "
                           "Use this whenever a caller asks what times are free.",
            "parameters": {
                "type": "object",
                "required": ["date"],
                "properties": {
                    "date": {"type": "string", "description": "The day as YYYY-MM-DD"}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_appointment",
            "description": "Book an appointment once you have confirmed the caller's name, "
                           "the date, and the time. Always confirm details aloud first.",
            "parameters": {
                "type": "object",
                "required": ["name", "date", "time"],
                "properties": {
                    "name": {"type": "string", "description": "The caller's full name"},
                    "date": {"type": "string", "description": "The day as YYYY-MM-DD"},
                    "time": {"type": "string", "description": "A slot like '10:00 AM'"},
                    "reason": {"type": "string", "description": "Reason for the visit"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Transfer the caller to a live team member for emergencies, "
                           "complaints, or anything outside your scope.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why the transfer is needed"}
                },
            },
        },
    },
]

# Map tool names to the real functions in tools.py.
TOOLS_IMPL = {
    "check_availability": lambda i: tools.check_availability(i["date"]),
    "book_appointment": lambda i: tools.book_appointment(
        i["name"], i["date"], i["time"], i.get("reason", "a general visit")
    ),
    "transfer_to_human": lambda i: tools.transfer_to_human(i.get("reason", "a general question")),
}


# ---------------------------------------------------------------------------
# The agent loop. Returns the final spoken text WITHOUT appending it;
# main() appends the reply so fallbacks stay consistent.
# ---------------------------------------------------------------------------
def agent_turn(history):
    today = datetime.date.today().isoformat()
    # The system prompt is rebuilt each turn so the injected "today's date" is fresh;
    # OpenAI caches long, identical prompt prefixes automatically (no flags needed).
    system = PERSONA + f"\n\nToday's date is {today}."
    # Snapshot so a mid-loop failure can be rolled back: resilient() retries
    # agent_turn(history) on the SAME list, and we must not feed it a half-mutated
    # history (e.g. an assistant tool_calls message whose tool replies never landed).
    base_len = len(history)
    try:
        return _run_turn(history, system)
    except Exception:
        del history[base_len:]
        raise


def _run_turn(history, system):
    while True:
        resp = client.chat.completions.create(
            model=PRIMARY_MODEL,
            max_completion_tokens=200,
            messages=[{"role": "system", "content": system}] + history,
            tools=TOOLS,
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            engine.play_interruptible("One moment.")        # cover tool latency, don't go silent
            # Echo the assistant's tool-call message back into history verbatim, then
            # answer each call with a matching "tool" message (OpenAI requires the pair).
            history.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    out = TOOLS_IMPL[tc.function.name](args)
                except Exception as e:
                    out = f"Tool error: {e}"
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(out),
                })
            continue
        return msg.content or ""


def trim_history(history, max_msgs=MAX_TURNS * 2):
    """Cap memory, but only cut at a clean caller turn so we never split a tool exchange."""
    if len(history) <= max_msgs:
        return history
    cut = len(history) - max_msgs
    while cut < len(history):
        m = history[cut]
        if m["role"] == "user" and isinstance(m["content"], str):
            return history[cut:]
        cut += 1
    return history


def main():
    utterances = queue.Queue()
    # Measure speaker-vs-headphone echo coupling and auto-tune barge-in BEFORE the
    # listener grabs the mic (calibration needs exclusive use of the audio devices).
    engine.calibrate_echo()
    listener_thread = threading.Thread(target=engine.listener, args=(utterances.put,), daemon=True)
    listener_thread.start()

    history = []
    engine.play_interruptible(
        f"Thanks for calling {tools.BUSINESS['name']}. How can I help you today?"
    )
    print("Receptionist live. Speak any time — even while the receptionist is talking.")
    print("Headphones recommended (so the mic doesn't hear the receptionist). Ctrl+C to quit.\n")

    try:
        while True:
            audio = utterances.get()                 # blocks until a full utterance arrives
            with engine.timed("asr"):                # time ONLY transcription, not the wait
                text = engine.transcribe(audio)
            if not text:
                continue
            # Content guard (layer 2): even if a transcript got past the listener's
            # acoustic gate, drop it if it closely matches what we just said — it's our
            # own voice echoing back, not a caller.
            if engine.is_self_echo(text):
                if engine.RX_DEBUG:
                    engine.log(f"[caller?] tts_now={engine.agent_speaking.is_set()} "
                               f"rms={engine._rms(audio):.4f} text={text!r} -> DROP (self-echo)")
                continue
            if engine.RX_DEBUG:
                engine.log(f"[caller?] tts_now={engine.agent_speaking.is_set()} "
                           f"rms={engine._rms(audio):.4f} text={text!r} -> ACCEPT")
            print("Caller:", text)
            history = trim_history(history)
            history.append({"role": "user", "content": text})
            with engine.timed("agent_turn"):
                reply = engine.resilient(
                    lambda: agent_turn(history),
                    lambda: "I'm sorry, could you say that again?",
                    on_total_failure="Let me transfer you to a colleague.",
                )
            history.append({"role": "assistant", "content": reply})
            print("Receptionist:", reply, "\n")
            # Clear any barge-in left over from capturing this turn's utterance so
            # it can't cut off the reply before it starts. Live barge-in during
            # this playback is still honored: the listener re-sets the event.
            engine.barge_in.clear()
            engine.play_interruptible(reply)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown. The listener is a daemon thread blocked in PortAudio's
        # sd.InputStream; if the interpreter just exits, that native stream gets torn
        # down from under the still-running thread and segfaults. So signal it to stop,
        # let it close its InputStream (via the `with` block), and join before exiting.
        print()
        engine.stop.set()
        listener_thread.join(timeout=2.0)
        engine.latency_summary()
        print("Goodbye.")


if __name__ == "__main__":
    main()
