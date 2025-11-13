"""
db/__init__.py
Helper script to initialize the database and optionally seed demo data.

Usage:
    python -m db.db_init         # creates tables
    python -m db.db_init --seed  # creates tables and seeds one demo registered vehicle
"""

import argparse
from db.session import create_all_tables, SessionLocal
from db.models import RegisteredVehicle
from utils.logger import get_logger

logger = get_logger("db.db_init")


def seed_demo_data(session):
    """Insert a sample registered vehicle if not present."""
    example_plate = "TN09AB1234"
    existing = session.query(RegisteredVehicle).filter_by(plate_text=example_plate).first()
    if existing:
        logger.info("Demo registered vehicle already exists: %s", existing)
        return existing
    demo = RegisteredVehicle(
        plate_text=example_plate,
        owner_name="Demo Owner",
        phone_number="+911234567890"
    )
    session.add(demo)
    session.commit()
    logger.info("Inserted demo RegisteredVehicle: %s", demo)
    return demo


def main(seed: bool = False):
    logger.info("Creating database tables...")
    create_all_tables()
    logger.info("Done creating tables.")
    if seed:
        session = SessionLocal()
        try:
            seed_demo_data(session)
        finally:
            session.close()
    logger.info("DB init complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize DB and optionally seed demo data.")
    parser.add_argument("--seed", action="store_true", help="Seed demo data after creating tables.")
    args = parser.parse_args()
    main(seed=args.seed)
