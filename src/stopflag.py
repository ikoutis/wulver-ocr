"""Cooperative stop for preemption and the wall clock.

SLURM sends SIGUSR1 before the time limit (and when a qos=low job is
preempted with a grace period); slurm/requeue_lib.sh forwards it to the
pipeline. The handler only sets an Event. Every unit of work checks it
before it starts (a page read, a block review, a figure description, a
retry), so after the signal no new request is sent, the ones in flight
finish, every finished page is saved, and the CLI exits 85.

This module deliberately imports nothing heavy, so run_ocr can install the
handler before importing PIL/httpx (a signal that arrives during start-up
would otherwise kill the interpreter, see slurm/requeue_lib.sh).
"""

from __future__ import annotations

import signal
import sys
import threading

STOP = threading.Event()


class Stopped(Exception):
    """A stop was requested. Not an error: the work item was not started."""


def check_stop() -> None:
    if STOP.is_set():
        raise Stopped()


def _on_signal(signum, _frame):
    if not STOP.is_set():
        print(f"[signal {signum}] no new requests; finishing those in flight, "
              "saving, then exiting 85", file=sys.stderr, flush=True)
    STOP.set()


def install_signal_handlers() -> None:
    signal.signal(signal.SIGUSR1, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
