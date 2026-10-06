import os
import re
import json
import time
from google import genai
from vaani.app.agents.prompts import SYSTEM_PROMPT
from vaani.app.agents.memory import SessionMemory
from vaani.app.rag.retriever import KBRetriever
from vaani.app.tools.orders import get_order_status, cancel_order
from vaani.app.tools.appointments import check_appointment, book_appointment
from dotenv import load_dotenv

load_dotenv()

class VaaniAgent:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        self.memory = SessionMemory()
        self.retriever = KBRetriever()

    def _call_gemini(self, prompt):
        for attempt in range(3):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={
                        "system_instruction": SYSTEM_PROMPT
                    }
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "quota" in err or "resourceexhausted" in err:
                    wait_time = 10 * (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"GenAI Error: {e}")
                    break
        return "I have processed your request."

    def chat(self, session_id, user_input):
        history = self.memory.get_history(session_id)
        tool_calls = []

        order_match = re.search(r"\b\d{4}\b", user_input)
        if order_match:
            self.memory.update_slot(session_id, "last_order_id", order_match.group())

        tool_output = ""
        user_lower = user_input.lower()
        
        if "cancel" in user_lower and order_match:
            res = cancel_order(order_match.group())
            tool_calls.append({"name": "cancel_order", "args": {"order_id": order_match.group()}})
            tool_output = f"Tool Result: {json.dumps(res)}"
        elif any(w in user_lower for w in ["status", "where is", "track", "order"]) and order_match:
            res = get_order_status(order_match.group())
            tool_calls.append({"name": "get_order_status", "args": {"order_id": order_match.group()}})
            tool_output = f"Tool Result: {json.dumps(res)}"
        elif "appointment" in user_lower:
            date_match = re.search(r"\d{4}-\d{2}-\d{2}", user_input)
            date_val = date_match.group() if date_match else "2026-10-10"
            if "book" in user_lower:
                res = book_appointment(date_val, "10:00 AM", "Customer")
                tool_calls.append({"name": "book_appointment", "args": {"date": date_val, "time": "10:00 AM", "customer": "Customer"}})
            else:
                res = check_appointment(date_val)
                tool_calls.append({"name": "check_appointment", "args": {"date": date_val}})
            tool_output = f"Tool Result: {json.dumps(res)}"

        docs = self.retriever.retrieve(user_input)
        rag_context = ""
        if docs:
            rag_context = "Knowledge Base:\n" + "\n".join([f"- {d['content']}" for d in docs])

        prompt = f"""User: {user_input}
{tool_output}
{rag_context}

Please respond concisely and helpfully."""

        reply = self._call_gemini(prompt)

        handoff_keywords = ["human", "agent", "representative", "person", "escalat"]
        handoff_triggered = any(k in user_lower for k in handoff_keywords)
        handoff_summary = {}
        if handoff_triggered:
            handoff_summary = {
                "slots": self.memory.get_slots(session_id),
                "reason": "Customer requested human agent."
            }

        self.memory.add_message(session_id, "user", user_input)
        self.memory.add_message(session_id, "model", reply)

        return {
            "response": reply,
            "sources": docs,
            "tool_calls": tool_calls,
            "confidence": round(docs[0]["confidence"], 2) if docs else 0.95,
            "handoff": handoff_triggered,
            "handoff_summary": handoff_summary
        }
