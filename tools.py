"""
tools.py — the receptionist's real-world capabilities.

This is one of the two "slots" that define the project (the other is the
persona). A different voice agent would swap this file for its own tools and
keep engine.py untouched.

The appointment "database" is a plain JSON file so you can open it and see
exactly what got booked. In production this becomes a real database.
"""

import os
import json

# ---- Business facts. The persona reads these so the agent can answer FAQs. ----
BUSINESS = {
    "name": "Bright Smile Dental",
    "hours": "Monday to Friday, nine in the morning to five in the evening",
    "address": "200 Oak Street, Springfield",
    "phone": "555-0142",
    "services": "cleanings, check-ups, fillings, and whitening",
}

# Bookable times in a day.
SLOTS = ["9:00 AM", "10:00 AM", "11:00 AM", "1:00 PM", "2:00 PM", "3:00 PM", "4:00 PM"]

DB_PATH = os.environ.get("APPOINTMENTS_DB", "appointments.json")


def _load():
    if os.path.exists(DB_PATH):
        with open(DB_PATH) as f:
            return json.load(f)
    return {}


def _save(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=2)


def check_availability(date):
    """Return the open time slots for a given YYYY-MM-DD date."""
    booked = {a["time"] for a in _load().get(date, [])}
    free = [s for s in SLOTS if s not in booked]
    if not free:
        return f"There are no openings on {date}."
    return f"Open times on {date} are: " + ", ".join(free) + "."


def book_appointment(name, date, time, reason="a general visit"):
    """Book a slot. Refuses double-booking; persists to the JSON store."""
    db = _load()
    day = db.setdefault(date, [])
    if any(a["time"] == time for a in day):
        return f"Sorry, {time} on {date} is already taken. Please pick another time."
    day.append({"name": name, "time": time, "reason": reason})
    _save(db)
    return f"Done. {name} is booked for {reason} on {date} at {time}."


def transfer_to_human(reason="a general question"):
    """Signal a transfer to a live team member. In production this routes the call."""
    return f"Transferring you to a team member now regarding {reason}. Please hold."
