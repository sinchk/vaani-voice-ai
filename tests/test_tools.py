import pytest

from tests.conftest import FUTURE_DAY
from vaani.tools.appointments import BUSINESS_SLOTS, normalize_time


class TestOrders:
    def test_status_found(self, order_service):
        r = order_service.get_order_status("1042")
        assert r["ok"] and r["order"]["status"] == "Shipped"

    def test_status_accepts_hash_prefix_and_ints(self, order_service):
        assert order_service.get_order_status("#1042")["ok"]
        assert order_service.get_order_status(1042)["ok"]

    def test_status_not_found(self, order_service):
        r = order_service.get_order_status("9999")
        assert not r["ok"] and "not found" in r["error"]

    def test_cancel_processing_order_persists(self, order_service):
        assert order_service.cancel_order("1043")["ok"]
        assert order_service.get_order_status("1043")["order"]["status"] == "Cancelled"

    def test_cancel_shipped_order_refused(self, order_service):
        r = order_service.cancel_order("1042")
        assert not r["ok"] and "Shipped" in r["error"]
        assert order_service.get_order_status("1042")["order"]["status"] == "Shipped"

    def test_cancel_twice_reports_already_cancelled(self, order_service):
        assert "already cancelled" in order_service.cancel_order("1044")["error"]

    def test_cancel_unknown_order(self, order_service):
        assert not order_service.cancel_order("9999")["ok"]

    def test_seed_is_not_modified(self, order_service):
        order_service.cancel_order("1043")
        assert '"Processing"' in order_service.store.seed_path.read_text()


class TestNormalizeTime:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("10am", "10:00 AM"),
            ("10:00 am", "10:00 AM"),
            ("2 PM", "02:00 PM"),
            ("14:00", "02:00 PM"),
            ("12pm", "12:00 PM"),
            ("0:00", "12:00 AM"),
            ("10:00 AM", "10:00 AM"),
        ],
    )
    def test_valid(self, raw, expected):
        assert normalize_time(raw) == expected

    @pytest.mark.parametrize("raw", ["", "noon", "10:30 AM", "13 PM", "25:00"])
    def test_invalid(self, raw):
        assert normalize_time(raw) is None


class TestAppointments:
    def test_check_lists_open_and_booked(self, appointment_service):
        r = appointment_service.check_appointment(FUTURE_DAY)
        assert r["ok"] and r["booked_slots"] == ["11:00 AM"]
        assert "11:00 AM" not in r["available_slots"]
        assert len(r["available_slots"]) == len(BUSINESS_SLOTS) - 1

    def test_check_rejects_bad_date(self, appointment_service):
        assert not appointment_service.check_appointment("10/10/2030")["ok"]

    def test_book_then_slot_is_taken(self, appointment_service):
        assert appointment_service.book_appointment(FUTURE_DAY, "10am", "Meera")["ok"]
        assert "10:00 AM" in appointment_service.check_appointment(FUTURE_DAY)["booked_slots"]
        again = appointment_service.book_appointment(FUTURE_DAY, "10:00 AM", "Someone")
        assert not again["ok"] and "already booked" in again["error"]

    @pytest.mark.parametrize(
        ("date", "time", "customer", "fragment"),
        [
            ("2020-01-01", "10:00 AM", "A", "past"),
            ("not-a-date", "10:00 AM", "A", "YYYY-MM-DD"),
            (FUTURE_DAY, "9:00 PM", "A", "not a valid slot"),
            (FUTURE_DAY, "10:00 AM", "  ", "name is required"),
        ],
    )
    def test_book_validation(self, appointment_service, date, time, customer, fragment):
        r = appointment_service.book_appointment(date, time, customer)
        assert not r["ok"] and fragment in r["error"]


class TestRegistry:
    def test_declarations_are_complete(self, registry):
        names = {d["name"] for d in registry.declarations()}
        assert names == {
            "get_order_status",
            "cancel_order",
            "check_appointment",
            "book_appointment",
            "request_human_handoff",
        }
        for d in registry.declarations():
            assert d["description"] and d["parameters"]["required"]

    def test_execute_dispatches(self, registry):
        assert registry.execute("get_order_status", {"order_id": "1042"})["ok"]

    def test_unknown_tool_is_returned_not_raised(self, registry):
        assert not registry.execute("drop_tables", {})["ok"]

    def test_bad_arguments_are_returned_not_raised(self, registry):
        r = registry.execute("book_appointment", {"date": FUTURE_DAY})
        assert not r["ok"] and "Invalid arguments" in r["error"]
