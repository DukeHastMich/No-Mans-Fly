"""Bound UI backlog without dropping errors or intervention/event messages."""
import queue
import threading


class SnapshotQueue:
    def __init__(self):
        self.events = queue.SimpleQueue()
        self.lock = threading.Lock()
        self.latest = None

    def put(self, item):
        if item[0] == 'snapshot':
            with self.lock:
                self.latest = item
        else:
            self.events.put(item)

    def get_nowait(self):
        try:
            return self.events.get_nowait()
        except queue.Empty:
            with self.lock:
                if self.latest is None:
                    raise queue.Empty
                item, self.latest = self.latest, None
                return item
