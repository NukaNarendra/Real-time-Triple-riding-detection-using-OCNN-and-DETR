# web_app/routes.py
from flask import Blueprint, render_template, request, redirect, url_for, Response, flash, send_file
from werkzeug.utils import secure_filename
import os
import tempfile
from typing import Optional

from inference.pipeline import InferencePipeline
from db.session import SessionLocal
from db.models import Violation
from utils.logger import get_logger

bp = Blueprint("main", __name__)
logger = get_logger("web.routes")

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".flv"}


# ----------- Helper functions -----------

def allowed_file(filename: str) -> bool:
    """Return True if file extension is allowed."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXT


def _parse_video_source(src: Optional[str]):
    """
    Convert numeric strings like '0' -> int(0) so OpenCV opens camera devices.
    Leave real URLs / file paths as strings.
    Also supports 'camera:0' syntax.
    """
    if src is None:
        return None
    s = str(src).strip()
    # plain digits -> camera index
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return s
    # camera:0 style -> 0
    if s.lower().startswith("camera:"):
        tail = s[7:].strip()
        if tail.isdigit():
            try:
                return int(tail)
            except Exception:
                return s
    # otherwise keep original (file path, http/rtsp URL, etc.)
    return s


def stream_generator(video_source: str):
    """
    Yields frames as byte-mjpeg to the UI stream.
    Accepts:
      - numeric device index passed as string (e.g. "0")
      - file path (absolute or temp)
      - http/rtsp/mjpeg stream URL
    """
    parsed = _parse_video_source(video_source)
    logger.info("[WEB STREAM] Starting inference on: %s (parsed=%s)", video_source, parsed)
    pipeline = InferencePipeline(detector_device="cpu")  # CPU by default; change if GPU available
    for frame_bytes in pipeline.process_video(parsed):
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# ----------- Routes -----------

@bp.route("/", methods=["GET", "POST"])
def index():
    """Upload video or give URL."""
    if request.method == "POST":
        file = request.files.get("video_file")
        url = request.form.get("video_url", "").strip()

        # Case 1 — file upload
        if file and file.filename != "":
            filename = secure_filename(file.filename)

            if not allowed_file(filename):
                flash("❌ Unsupported file format", "danger")
                return redirect(url_for("main.index"))

            tmp_path = os.path.join(tempfile.gettempdir(), filename)
            file.save(tmp_path)

            logger.info("[UPLOAD] File saved to temp: %s", tmp_path)
            # pass local file path directly
            return redirect(url_for("main.stream", src=tmp_path))

        # Case 2 — video URL or camera index (e.g. "0")
        elif url:
            # allow camera index or full URL
            return redirect(url_for("main.stream", src=url))

        flash("⚠️ Provide a video file or URL", "warning")
        return redirect(url_for("main.index"))

    return render_template("index.html")


@bp.route("/stream")
def stream():
    """Returns real-time MJPEG stream."""
    src = request.args.get("src")
    if not src:
        return "Missing video source", 400

    return Response(stream_generator(src),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/violations")
def violations():
    """Display saved violations from DB."""
    session = SessionLocal()
    try:
        records = (
            session.query(Violation)
            .order_by(Violation.id.desc())
            .limit(30)
            .all()
        )
    finally:
        session.close()

    return render_template("violations.html", violations=records)


@bp.route("/download")
def download():
    """Download evidence image by path (pass ?file=/abs/path/to/file.jpg)."""
    path = request.args.get("file")

    if path and os.path.exists(path):
        # send_file handles correct mimetype/headers
        return send_file(path, as_attachment=True)

    return "File not found", 404
