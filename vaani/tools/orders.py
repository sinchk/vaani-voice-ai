"""Order lookup and cancellation."""

from __future__ import annotations

from typing import Any

from vaani.tools.store import JsonStore

# Statuses from which an order can still be cancelled.
CANCELLABLE = {"Processing", "Placed"}


def normalize_order_id(order_id: Any) -> str:
    return str(order_id).strip().lstrip("#")


class OrderService:
    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        order_id = normalize_order_id(order_id)
        for order in self.store.load():
            if order["order_id"] == order_id:
                return {"ok": True, "order": order}
        return {"ok": False, "error": f"Order {order_id} was not found."}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        order_id = normalize_order_id(order_id)
        with self.store.lock:
            orders = self.store.load()
            for order in orders:
                if order["order_id"] != order_id:
                    continue
                if order["status"] == "Cancelled":
                    return {"ok": False, "error": f"Order {order_id} is already cancelled."}
                if order["status"] not in CANCELLABLE:
                    return {
                        "ok": False,
                        "error": f"Order {order_id} is {order['status']} and can no longer be "
                        "cancelled. It can be returned under the return policy instead.",
                    }
                order["status"] = "Cancelled"
                self.store.save(orders)
                return {"ok": True, "message": f"Order {order_id} has been cancelled.", "order": order}
        return {"ok": False, "error": f"Order {order_id} was not found."}
