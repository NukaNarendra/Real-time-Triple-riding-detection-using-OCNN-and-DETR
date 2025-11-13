#!/usr/bin/env python3
"""
scripts/run_inference_server.py

Run the Flask inference server. This script initializes DB (if needed) and launches
the Flask app. Intended for development use.

Usage:
    python scripts/run_inference_server.py --host 0.0.0.0 --port 5000
"""
import argparse
import os
from web_app.app import create_app
from db.__init__ import main as db_init_main
from utils.logger import get_logger

logger = get_logger("scripts.run_server")

def ensure_db(seed: bool = False):
    try:
        db_init_main(seed)
    except Exception as e:
        logger.exception("DB init failed: %s", e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--seed-db", action="store_true", help="Seed DB with demo data")
    args = parser.parse_args()
    ensure_db(seed=args.seed_db)
    app = create_app()
    logger.info("Starting Flask app on %s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    main()
