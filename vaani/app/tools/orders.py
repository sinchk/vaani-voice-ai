import json
import os

ORDERS_FILE = "vaani/data/orders.json"

def load_orders():
    if os.path.exists(ORDERS_FILE):
        with open(ORDERS_FILE, "r") as f:
            return json.load(f)
    return []

def save_orders(orders):
    with open(ORDERS_FILE, "w") as f:
        json.dump(orders, f, indent=2)

def get_order_status(order_id: str):
    orders = load_orders()
    for order in orders:
        if order["order_id"] == order_id:
            return order
    return {"error": f"Order {order_id} not found."}

def cancel_order(order_id: str):
    orders = load_orders()
    for order in orders:
        if order["order_id"] == order_id:
            if order["status"] == "Shipped":
                return {"error": "Cannot cancel order already shipped."}
            order["status"] = "Cancelled"
            save_orders(orders)
            return {"success": f"Order {order_id} cancelled successfully."}
    return {"error": f"Order {order_id} not found."}
