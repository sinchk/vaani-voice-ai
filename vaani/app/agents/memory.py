class SessionMemory:
    def __init__(self):
        self.sessions = {}

    def get_history(self, session_id):
        if session_id not in self.sessions:
            self.sessions[session_id] = {"messages": [], "slots": {}}
        return self.sessions[session_id]

    def add_message(self, session_id, role, content):
        history = self.get_history(session_id)
        history["messages"].append({"role": role, "parts": [content]})

    def update_slot(self, session_id, key, value):
        history = self.get_history(session_id)
        history["slots"][key] = value

    def get_slots(self, session_id):
        return self.get_history(session_id)["slots"]
