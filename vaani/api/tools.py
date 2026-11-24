"""Direct access to the business tools (useful for debugging and the demo UI)."""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from vaani.api.deps import get_tools
from vaani.tools import ToolRegistry

router = APIRouter(prefix="/tools")


class BookingRequest(BaseModel):
    date: str = Field(description="YYYY-MM-DD")
    time: str = Field(description="e.g. '10:00 AM'")
    customer: str


@router.get("/orders/{order_id}", summary="Get order status")
def order_status(order_id: str, tools: ToolRegistry = Depends(get_tools)) -> dict[str, Any]:
    return tools.orders.get_order_status(order_id)


@router.post("/orders/{order_id}/cancel", summary="Cancel an order")
def order_cancel(order_id: str, tools: ToolRegistry = Depends(get_tools)) -> dict[str, Any]:
    return tools.orders.cancel_order(order_id)


@router.get("/appointments", summary="Check appointment availability for a date")
def appointments_check(date: str, tools: ToolRegistry = Depends(get_tools)) -> dict[str, Any]:
    return tools.appointments.check_appointment(date)


@router.post("/appointments", summary="Book an appointment")
def appointments_book(req: BookingRequest, tools: ToolRegistry = Depends(get_tools)) -> dict[str, Any]:
    return tools.appointments.book_appointment(req.date, req.time, req.customer)
