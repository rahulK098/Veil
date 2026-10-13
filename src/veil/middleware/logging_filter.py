"""``logging.Filter`` that runs every record through the storage policy before it is emitted.

This is the "surface most teams forget": attach it once to a logger or handler and every
message (after ``%`` formatting) goes through the ``logs`` destination. Blocked records are
suppressed entirely rather than emitted half-redacted.
"""

from __future__ import annotations

import logging

from veil.middleware.guard import Guard
from veil.policy.schema import Destination


class VeilLogFilter(logging.Filter):
    def __init__(self, guard: Guard, *, request_id_attr: str | None = "request_id") -> None:
        super().__init__(name="veil")
        self._guard = guard
        self._request_id_attr = request_id_attr

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        rid = getattr(record, self._request_id_attr, None) if self._request_id_attr else None
        scrubbed = self._guard.for_storage(
            message, Destination.LOGS, request_id=rid if isinstance(rid, str) else None
        )
        if scrubbed is None:
            return False  # policy blocked this line; drop it
        record.msg = scrubbed
        record.args = ()
        return True
