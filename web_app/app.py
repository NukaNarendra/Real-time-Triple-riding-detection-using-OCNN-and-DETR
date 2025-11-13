# web_app/app.py
from flask import Flask
from web_app.routes import bp
from utils.logger import get_logger
import os

logger = get_logger("web.app")

def create_app():
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = os.getenv("FLASK_SECRET", "dev-secret-key")
    app.register_blueprint(bp)
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, threaded=True)
