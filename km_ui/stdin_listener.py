# ui/stdin_listener.py
"""
Non-blocking stdin listener for ID selection and simple commands.

Usage:
    from ui.stdin_listener import start_stdin_listener
    q = start_stdin_listener(prompt="> ")
    # in your main loop:
    try:
        msg = q.get_nowait()
        # msg is a stripped string: "23", "n", "q", "stop", "go", etc.
    except queue.Empty:
        pass

Design notes:
- Runs a daemon thread so it won't block program exit.
- Works on Linux/macOS/Windows (uses blocking readline in a background thread).
- Sanitizes inputs (strip whitespace) and drops empty lines or comments (# ...).
- Provides stop() to shut down cleanly if desired.
"""

from __future__ import annotations
import sys
import threading
import queue
from typing import Optional

__all__ = [
    "start_stdin_listener",
    "stop_stdin_listener",
    "is_running",
    "get_queue",
]

# --- module state ---
_q: Optional[queue.Queue[str]] = None
_t: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_prompt: str = ""


def _reader():
    """Background thread: read lines and enqueue."""
    global _q, _stop_event, _prompt
    # If stdin is not a TTY (piped), we still read lines until EOF.
    while _stop_event and not _stop_event.is_set():
        try:
            # Show prompt only if interactive
            if _prompt and sys.stdin and sys.stdin.isatty():
                sys.stdout.write(_prompt)
                sys.stdout.flush()

            line = sys.stdin.readline()
            if not line:
                # EOF or stream closed
                break

            msg = line.strip()
            if not msg or msg.startswith("#"):
                continue  # ignore empty/comment

            # Push to queue
            if _q:
                _q.put(msg)
        except Exception:
            # Swallow exceptions so the listener can't crash the app.
            # Consider logging here if you have a logger.
            continue


def start_stdin_listener(prompt: str = "") -> "queue.Queue[str]":
    """
    Start the stdin listener thread (idempotent). Returns the message queue.

    Typical commands (handled by your pipeline/target manager):
      - <int> : promote that detection ID to active target
      - n     : clear target (neutral)
      - q     : quit program
      - stop  : emergency stop (if wired)
      - go    : clear E-stop (if wired)
    """
    global _q, _t, _stop_event, _prompt
    if _t and _t.is_alive() and _q:
        return _q

    _prompt = str(prompt or "")
    _q = queue.Queue()
    _stop_event = threading.Event()

    _t = threading.Thread(target=_reader, name="stdin-listener", daemon=True)
    _t.start()
    return _q


def stop_stdin_listener(timeout: Optional[float] = 0.5) -> None:
    """Signal the thread to stop and wait briefly."""
    global _t, _stop_event
    if _stop_event:
        _stop_event.set()
    if _t and _t.is_alive():
        _t.join(timeout=timeout or 0.0)


def is_running() -> bool:
    """Return True if the listener thread is alive."""
    return bool(_t and _t.is_alive())


def get_queue() -> Optional["queue.Queue[str]"]:
    """Return the queue if the listener was started, else None."""
    return _q


if __name__ == "__main__":
    # Simple demo: echo lines until 'q'
    q = start_stdin_listener(prompt="> ")
    print("[stdin_listener] Type lines (q to quit).")
    import time
    try:
        while True:
            try:
                m = q.get_nowait()
                print(f"[recv] {m!r}")
                if m.lower() == "q":
                    break
            except queue.Empty:
                time.sleep(0.02)
    finally:
        stop_stdin_listener()
        print("[stdin_listener] stopped.")

