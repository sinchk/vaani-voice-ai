import json
import os

APPOINTMENTS_FILE = "vaani/data/appointments.json"

def load_appointments():
    if os.path.exists(APPOINTMENTS_FILE):
        with open(APPOINTMENTS_FILE, "r") as f:
            return json.load(f)
    return []

def save_appointments(apps):
    with open(APPOINTMENTS_FILE, "w") as f:
        json.dump(apps, f, indent=2)

def check_appointment(date: str):
    """Check appointment slots for a given date (YYYY-MM-DD)."""
    apps = load_appointments()
    matches = [a for a in apps if a["date"] == date]
    return {"appointments": matches}

def book_appointment(date: str, time: str, customer: str):
    """Book an appointment for a customer on a date and time."""
    apps = load_appointments()
    for a in apps:
        if a["date"] == date and a["time"] == time:
            if a["status"] == "Booked":
                return {"error": f"Slot at {time} on {date} is already booked."}
            a["status"] = "Booked"
            a["customer"] = customer
            save_appointments(apps)
            return {"success": f"Appointment booked for {customer} at {time} on {date}."}
    
    new_app = {"date": date, "time": time, "customer": customer, "status": "Booked"}
    apps.append(new_app)
    save_appointments(apps)
    return {"success": f"New appointment created and booked for {customer} at {time} on {date}."}
