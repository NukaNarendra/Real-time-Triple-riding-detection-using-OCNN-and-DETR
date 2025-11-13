"""
sms/notification_service.py

High-level NotificationService that picks the provider implementation based on
environment variable SMS_PROVIDER (default: mock). Exposes:

- send_message(phone, message, mark_violation_id=None)
- notify_violation(violation_id)

notify_violation:
    - looks up Violation by id; if violation.vehicle_plate exists and maps to a RegisteredVehicle with phone_number,
      it sends message to that phone and marks Violation.notification_sent when successful.
    - if no registration is found, it logs and returns False.
"""

import os
from utils.logger import get_logger
from db.session import SessionLocal
from db.models import Violation, RegisteredVehicle
from typing import Optional

logger = get_logger("sms.notification_service")

SMS_PROVIDER = os.getenv("SMS_PROVIDER", "mock").lower()

# dynamic import of adapters
if SMS_PROVIDER == "twilio":
    try:
        from sms.twilio_adapter import send_sms_twilio as _send_impl
    except Exception as e:
        logger.exception("Failed to import Twilio adapter: %s", e)
        _send_impl = None
else:
    try:
        from sms.mock_sms import send_sms as _send_impl
    except Exception as e:
        logger.exception("Failed to import mock SMS adapter: %s", e)
        _send_impl = None


class NotificationService:
    def __init__(self, provider_impl=None):
        self.send_impl = provider_impl or _send_impl
        if self.send_impl is None:
            raise RuntimeError("No SMS provider available. Configure SMS_PROVIDER and install dependencies.")

    def send_message(self, phone_number: str, message: str, mark_violation_id: Optional[int] = None) -> bool:
        try:
            ok = self.send_impl(phone_number, message, mark_violation_id)
            if ok:
                logger.info("Message sent to %s", phone_number)
                return True
            else:
                logger.warning("Message NOT sent to %s", phone_number)
                return False
        except Exception as e:
            logger.exception("Exception while sending message: %s", e)
            return False

    def notify_violation(self, violation_id: int, extra_text: Optional[str] = None) -> bool:
        session = SessionLocal()
        try:
            v = session.query(Violation).filter(Violation.id == int(violation_id)).first()
            if not v:
                logger.warning("Violation id %s not found", violation_id)
                return False
            if v.notification_sent:
                logger.info("Violation %s already notified", violation_id)
                return True
            plate = v.vehicle_plate
            if not plate:
                logger.warning("Violation %s has no recognized plate text; cannot notify", violation_id)
                return False
            reg = session.query(RegisteredVehicle).filter(RegisteredVehicle.plate_text == plate).first()
            if not reg:
                logger.warning("No registered vehicle found for plate %s", plate)
                return False
            phone = reg.phone_number
            if not phone:
                logger.warning("Registered vehicle %s has no phone number", plate)
                return False
            ts = v.timestamp.isoformat() if v.timestamp else "unknown time"
            msg = f"Traffic Alert: Triple riding detected for {plate} at {ts}."
            if extra_text:
                msg = msg + " " + extra_text
            ok = self.send_message(phone, msg, mark_violation_id=violation_id)
            if ok:
                # mark done in DB in adapter as well, but ensure transaction here
                v.notification_sent = True
                session.add(v)
                session.commit()
            return ok
        except Exception as e:
            logger.exception("Failed to notify for violation %s: %s", violation_id, e)
            return False
        finally:
            session.close()


# convenience singleton
default_notification_service = NotificationService()
