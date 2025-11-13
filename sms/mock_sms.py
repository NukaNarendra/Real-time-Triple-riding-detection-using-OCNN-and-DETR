"""
sms/mock_sms.py

Mock SMS provider for development:
- Logs the message
- Persists a NotificationLog entry in DB with mock=True
- Optionally sets the Violation.notification_sent flag when provided with a violation id
"""

from datetime import datetime
import logging
from db.session import SessionLocal
from db.models import NotificationLog, Violation
from utils.logger import get_logger

logger = get_logger("sms.mock_sms")


def send_sms(phone_number: str, message: str, mark_violation_id: int = None) -> bool:
    """
    Send a mock SMS: log it and write NotificationLog to DB.
    If mark_violation_id is provided, set violation.notification_sent = True.

    Returns True on success, False on failure.
    """
    try:
        logger.info("[MOCK SMS] To %s: %s", phone_number, message)
        session = SessionLocal()
        notif = NotificationLog(
            phone_number=str(phone_number),
            message=message,
            sent_at=datetime.utcnow(),
            mock=True
        )
        session.add(notif)
        if mark_violation_id is not None:
            v = session.query(Violation).filter(Violation.id == int(mark_violation_id)).first()
            if v:
                v.notification_sent = True
                session.add(v)
        session.commit()
        session.close()
        return True
    except Exception as e:
        logger.exception("Failed to write mock SMS to DB: %s", e)
        return False
