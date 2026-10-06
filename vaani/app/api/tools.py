from fastapi import APIRouter
from vaani.app.tools.orders import get_order_status, cancel_order
from vaani.app.tools.appointments import check_appointment, book_appointment

router = APIRouter()

@router.get("/tools/order/{order_id}")
async def order_status(order_id: str):
    return get_order_status(order_id)

@router.post("/tools/order/cancel/{order_id}")
async def order_cancel(order_id: str):
    return cancel_order(order_id)

@router.get("/tools/appointments")
async def appointments_check(date: str, time: str = None):
    return check_appointment(date, time)
