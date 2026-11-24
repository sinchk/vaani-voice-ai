"""Tool registry: one place that defines what the agent can do.

Each :class:`ToolSpec` carries the JSON-schema the LLM sees *and* the Python
callable that runs it, so the declaration and the implementation cannot drift.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from vaani.config import Settings, get_settings
from vaani.tools.appointments import AppointmentService
from vaani.tools.orders import OrderService
from vaani.tools.store import JsonStore

logger = logging.getLogger(__name__)

HANDOFF_TOOL = "request_human_handoff"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]

    def declaration(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_STR = "string"


class ToolRegistry:
    def __init__(self, orders: OrderService, appointments: AppointmentService) -> None:
        self.orders = orders
        self.appointments = appointments
        specs = [
            ToolSpec(
                "get_order_status",
                "Look up the status, items and delivery ETA of an order by its numeric order ID.",
                _obj({"order_id": {"type": _STR, "description": "Order ID, e.g. '1042'."}}, ["order_id"]),
                orders.get_order_status,
            ),
            ToolSpec(
                "cancel_order",
                "Cancel an order. Only call after the customer has clearly asked to cancel a specific "
                "order. Fails if the order has already shipped.",
                _obj({"order_id": {"type": _STR, "description": "Order ID, e.g. '1043'."}}, ["order_id"]),
                orders.cancel_order,
            ),
            ToolSpec(
                "check_appointment",
                "List available and booked appointment slots for a date.",
                _obj({"date": {"type": _STR, "description": "Date in YYYY-MM-DD format."}}, ["date"]),
                appointments.check_appointment,
            ),
            ToolSpec(
                "book_appointment",
                "Book an appointment slot. Requires date, time and the customer's name; ask the customer "
                "for any that are missing instead of guessing.",
                _obj(
                    {
                        "date": {"type": _STR, "description": "Date in YYYY-MM-DD format."},
                        "time": {"type": _STR, "description": "Slot start time, e.g. '10:00 AM'."},
                        "customer": {"type": _STR, "description": "Customer's name."},
                    },
                    ["date", "time", "customer"],
                ),
                appointments.book_appointment,
            ),
            ToolSpec(
                HANDOFF_TOOL,
                "Escalate the conversation to a human agent. Use when the customer asks for a person, "
                "is frustrated, or the request is outside what you can do.",
                _obj({"reason": {"type": _STR, "description": "Short reason for the escalation."}}, ["reason"]),
                lambda reason: {"ok": True, "message": "A human agent has been notified.", "reason": reason},
            ),
        ]
        self._specs = {s.name: s for s in specs}

    @classmethod
    def default(cls, settings: Settings | None = None) -> ToolRegistry:
        s = settings or get_settings()
        orders = JsonStore(s.seed_data_dir / "orders.json", s.state_dir / "orders.json")
        appointments = JsonStore(s.seed_data_dir / "appointments.json", s.state_dir / "appointments.json")
        return cls(OrderService(orders), AppointmentService(appointments))

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def declarations(self) -> list[dict[str, Any]]:
        return [s.declaration() for s in self._specs.values()]

    def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run a tool; never raises - failures are returned so the LLM can recover."""
        spec = self._specs.get(name)
        if spec is None:
            return {"ok": False, "error": f"Unknown tool '{name}'."}
        try:
            return spec.fn(**args)
        except TypeError as exc:  # missing / unexpected arguments from the model
            logger.warning("Bad arguments for %s: %s (%s)", name, args, exc)
            return {"ok": False, "error": f"Invalid arguments for {name}: {exc}"}
        except Exception:  # noqa: BLE001 - a tool bug must not crash the conversation
            logger.exception("Tool %s failed", name)
            return {"ok": False, "error": f"{name} failed unexpectedly."}
