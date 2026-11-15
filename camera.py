"""Opens a camera, an RTSP stream or a video file with OpenCV."""
import os

import cv2


def open_camera(source):
    """source: a camera number (0, 1, ...), an rtsp:// address, or a video file."""
    if str(source).isdigit():
        return cv2.VideoCapture(int(source))
    if str(source).startswith("rtsp"):
        # TCP gives fewer broken frames than the default UDP
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        return cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    return cv2.VideoCapture(source)