"""Reads settings from the .env file and builds the camera (RTSP) address."""
import os
import re
from urllib.parse import quote

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env(path=ENV_PATH):
    """Minimal KEY=VALUE parser. '#' only starts a comment at the beginning of a
    line, so passwords may contain '#'. Optional surrounding quotes are removed."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key, value)


def rtsp_url():
    """Build rtsp://user:pass@host:port/path from the RTSP_* settings."""
    load_env()
    host = os.environ.get("RTSP_HOST", "").strip()
    if not host:
        raise SystemExit("RTSP_HOST is not set. Copy .env.example to .env and fill it in.")
    user = os.environ.get("RTSP_USER", "")
    password = os.environ.get("RTSP_PASSWORD", "")
    port = os.environ.get("RTSP_PORT", "").strip() or "554"
    path = os.environ.get("RTSP_PATH", "").strip()
    if path and not path.startswith("/"):
        path = "/" + path
    auth = ""
    if user:
        # percent-encode so characters like @ : / # in the password cannot break the URL
        auth = quote(user, safe="") + (":" + quote(password, safe="") if password else "") + "@"
    return f"rtsp://{auth}{host}:{port}{path}"


def camera_source(arg=None):
    """Which camera to open: the given value, else CAMERA_SOURCE from .env, else the RTSP stream from .env."""
    load_env()
    return arg or os.environ.get("CAMERA_SOURCE") or rtsp_url()


def mask_url(url):
    """Hide the password so the URL is safe to print."""
    return re.sub(r"(rtsp://[^:/@]+:)[^@]*@", r"\1***@", str(url))