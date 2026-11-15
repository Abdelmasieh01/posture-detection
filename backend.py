"""Sends data to the backend. If the backend is down, events and aggregates wait and are sent later."""
import json
import time
import urllib.error
import urllib.request


class Backend:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.waiting = []   # (path, data) that could not be sent yet, oldest first
        self.next_try = 0   # when the backend was down: do not try again before this time

    def send_status(self, data):
        """Live status. If it fails we just skip it, a newer one comes soon."""
        if time.time() >= self.next_try:
            self._post("/api/status/", data)

    def send_later(self, path, data):
        """Events and aggregates must not get lost: put them on the waiting list, then send what is waiting."""
        self.waiting.append((path, data))
        self.waiting = self.waiting[-500:]  # never keep more than 500
        self.flush()

    def flush(self):
        """Send the waiting items in order. Stops at the first one that cannot be reached."""
        while self.waiting and time.time() >= self.next_try:
            path, data = self.waiting[0]
            if self._post(path, data) == "down":
                return
            self.waiting.pop(0)  # sent, or refused by the backend (sending it again would not help)

    def _post(self, path, data):
        """Returns 'ok', 'refused' (the backend said no) or 'down' (could not reach it)."""
        request = urllib.request.Request(self.url + path, data=json.dumps(data).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=2):
                return "ok"
        except urllib.error.HTTPError as error:
            if error.code < 500:
                print(f"Backend refused {path} ({error.code})")
                return "refused"
        except OSError:  # no connection, timeout, ...
            pass
        self.next_try = time.time() + 10  # wait 10 s before trying again
        print("Backend not reachable, trying again in 10 s")
        return "down"