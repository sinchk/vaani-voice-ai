"""Appointment availability and booking."""

from __future__ import annotations

import re
from datetime import date as Date
from datetime import datetime
from typing import Any

from vaani.tools.store import JsonStore

# Hourly slots, 10:00 AM - 5:00 PM (the last slot starts at 4:00 PM).
BUSINESS_SLOTS = ["10:00 AM", "11:00 AM", "12:00 PM", "01:00 PM", "02:00 PM", "03:00 PM", "04:00 PM"]


def normalize_time(value: str) -> str | None:
    """Parse '10am', '10:00 am', '14:00', '2 PM' ... into 'HH:00 AM/PM'; None if invalid."""
    text = str(value).strip().upper().replace(".", "")
    match = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(AM|PM)?", text)
    if not match:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2] or 0), match[3]
    if minute != 0:
        return None
    if meridiem is None:  # 24-hour input
        if not 0 <= hour <= 23:
            return None
        meridiem = "PM" if hour >= 12 else "AM"
        hour = hour % 12 or 12
    elif not 1 <= hour <= 12:
        return None
    return f"{hour:02d}:00 {meridiem}"


def parse_date(value: str) -> Date | None:
    try:
        return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


class AppointmentService:
    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def check_appointment(self, date: str) -> dict[str, Any]:
        if parse_date(date) is None:
            return {"ok": False, "error": "Date must be in YYYY-MM-DD format."}
        booked = {a["time"] for a in self.store.load() if a["date"] == date and a["status"] == "Booked"}
        return {
            "ok": True,
            "date": date,
            "available_slots": [s for s in BUSINESS_SLOTS if s not in booked],
            "booked_slots": sorted(booked),
        }

    def book_appointment(self, date: str, time: str, customer: str) -> dict[str, Any]:
        day = parse_date(date)
        if day is None:
            return {"ok": False, "error": "Date must be in YYYY-MM-DD format."}
        if day < Date.today():
            return {"ok": False, "error": f"{date} is in the past. Please choose a future date."}
        slot = normalize_time(time)
        if slot is None or slot not in BUSINESS_SLOTS:
            return {
                "ok": False,
                "error": f"'{time}' is not a valid slot. Available slots: {', '.join(BUSINESS_SLOTS)}.",
            }
        customer = (customer or "").strip()
        if not customer:
            return {"ok": False, "error": "A customer name is required to book an appointment."}

        with self.store.lock:
            records = self.store.load()
            for record in records:
                if record["date"] == date and record["time"] == slot:
                    if record["status"] == "Booked":
                        return {"ok": False, "error": f"The {slot} slot on {date} is already booked."}
                    record.update(status="Booked", customer=customer)
                    break
            else:
                records.append({"date": date, "time": slot, "customer": customer, "status": "Booked"})
            self.store.save(records)
        return {"ok": True, "message": f"Appointment booked for {customer} at {slot} on {date}."}
