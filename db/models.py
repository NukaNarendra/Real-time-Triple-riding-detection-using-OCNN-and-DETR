"""
db/models.py
SQLAlchemy ORM models for:
- RegisteredVehicle: registry of plate -> owner info
- Violation: detected triple riding event
- NotificationLog: record of sent notifications (mock/real)
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RegisteredVehicle(Base):
    __tablename__ = "registered_vehicles"

    id = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String(64), unique=True, nullable=False, index=True)
    owner_name = Column(String(128), nullable=True)
    phone_number = Column(String(32), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<RegisteredVehicle(id={self.id} plate={self.plate_text} owner={self.owner_name})>"


class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_plate = Column(String(64), nullable=True, index=True)
    track_id = Column(String(64), nullable=True)
    video_source = Column(String(256), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    num_riders = Column(Integer, nullable=False, default=0)
    ocr_confidence = Column(String(32), nullable=True)
    evidence_path_raw = Column(String(512), nullable=True)     # restricted raw evidence
    evidence_path_blurred = Column(String(512), nullable=True) # blurred copy for UI
    notification_sent = Column(Boolean, default=False)
    additional_info = Column(Text, nullable=True)

    def __repr__(self):
        return (
            f"<Violation(id={self.id} plate={self.vehicle_plate} riders={self.num_riders}"
            f" time={self.timestamp.isoformat()})>"
        )


class NotificationLog(Base):
    __tablename__ = "notification_logs"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(32), nullable=False)
    message = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    mock = Column(Boolean, default=True)

    def __repr__(self):
        return f"<NotificationLog(id={self.id} phone={self.phone_number} mock={self.mock})>"
