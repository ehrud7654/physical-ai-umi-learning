"""Serialize local run creation, cancellation and supervisor journal updates."""
from functools import wraps
from threading import RLock

_lock = RLock()


def synchronized_run_changes(method):
    @wraps(method)
    def wrapped(*args, **kwargs):
        with _lock:
            return method(*args, **kwargs)
    return wrapped
