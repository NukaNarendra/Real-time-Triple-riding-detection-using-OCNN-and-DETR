"""
sms/twilio_adapter.py

Twilio SMS adapter. Reads TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER
from environment variables.

Usage:
    from sms.twilio_adapter import send_sms_twilio
    send_sms_twilio('+911234567890', 'Your message', mark_violation_id=123)
"""

import os
from datetime import datetime
from typing import Optional
from utils.logger import get_logger

logger = get_logger("sms.twilio_adapter")

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

from db.session import SessionLocal
from db.models import NotificationLog, Violation

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")


def _client():
    if not TWILIO_AVAILABLE:
        raise RuntimeError("twilio package not installed. pip install twilio")
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        raise RuntimeError("Twilio credentials missing. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER env vars.")
    return Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)


def send_sms_twilio(phone_number: str, message: str, mark_violation_id: Optional[int] = None) -> bool:
    """
    Send SMS via Twilio and log NotificationLog in DB. Also marks Violation.notification_sent if provided.

    Returns True on success, False on failure.
    """
    try:
        client = _client()
        resp = client.messages.create(body=message, from_=TWILIO_FROM_NUMBER, to=phone_number)
        logger.info("Twilio sent message SID=%s to %s", getattr(resp, "sid", "unknown"), phone_number)
        session = SessionLocal()
        notif = NotificationLog(
            phone_number=str(phone_number),
            message=message,
            sent_at=datetime.utcnow(),
            mock=False
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
        logger.exception("Failed to send SMS via Twilio: %s", e)
        return False
