"""Receives commands from the backend over a WebSocket, and reports how they go.

Flow: the mobile app sends a command to the backend (POST), the backend pushes it over this
WebSocket to the Pi, and the Pi sends progress messages back the same way (the backend passes them to the app).
"""
import json
import time

from websockets.sync.client import connect


class CommandChannel:
    def __init__(self, url):
        self.url = url      # for example ws://localhost:8000/ws/device/
        self.ws = None      # the open connection, or None
        self.next_try = 0   # when the connection is down: do not try again before this time

    def get(self):
        """The next message from the backend as a dict, for example
        {"type": "command", "command": "calibrate_camera"}, or None when there is nothing new. Never waits."""
        if self.ws is None and not self._connect():
            return None
        try:
            message = self.ws.recv(timeout=0)  # timeout=0: do not wait, raise TimeoutError if nothing is there
        except TimeoutError:
            return None
        except Exception:  # the connection was closed or broke
            print("Command channel lost, trying again in 10 s")
            self._lost()
            return None
        try:
            return json.loads(message)
        except ValueError:
            return None  # not JSON, ignore it

    def reply(self, data):
        """Tells the backend how a command is going (it passes this on to the app)."""
        if self.ws is None:
            return
        try:
            self.ws.send(json.dumps(data))
        except Exception:
            self._lost()

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def _connect(self):
        if time.time() < self.next_try:
            return False
        try:
            # legacy=True gives us one connection we can keep open while the loop runs
            self.ws = connect(self.url, open_timeout=2, legacy=True)
            print("Command channel connected")
            return True
        except Exception:
            print("Command channel not reachable, trying again in 10 s")
            self.next_try = time.time() + 10
            return False

    def _lost(self):
        self.ws = None
        self.next_try = time.time() + 10