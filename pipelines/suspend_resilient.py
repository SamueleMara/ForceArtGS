#!/usr/bin/env python
"""Run a long GPU command so that a laptop suspend (lid closed) pauses it instead of killing it.

On this machine a CUDA process that is running when the laptop suspends never recovers after resume: its main thread
spins at 100% CPU, the GPU stays idle and the log stops (verified against `journalctl` "Lid closed" events). This
wrapper keeps suspend-on-lid-close untouched and instead:

  1. detects the resume (CLOCK_BOOTTIME advances during suspend, CLOCK_MONOTONIC does not),
  2. kills the frozen command's whole process group,
  3. starts the same command again.

pipelines/joint_aware_reconstruction.py skips finished stages and resumes every ns-train from its newest intact
checkpoint (saved every 2000 steps), so a suspend costs at most one checkpoint interval of training.

    python pipelines/suspend_resilient.py -- python pipelines/joint_aware_reconstruction.py --scene ... --stage all

Ctrl+C / SIGTERM / SIGHUP (tmux kill-session) are forwarded to the command. SIGUSR1 simulates a resume (test hook).
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time


def suspended_seconds() -> float:
    """Total time the system has spent suspended since boot."""
    return time.clock_gettime(time.CLOCK_BOOTTIME) - time.monotonic()


def log(msg: str) -> None:
    print(f"[suspend-resilient {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def stop_group(p: subprocess.Popen, sig=signal.SIGTERM, grace: float = 30.0) -> None:
    try:
        os.killpg(p.pid, sig)
    except ProcessLookupError:
        return
    try:
        p.wait(grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-suspend", type=float, default=10.0, help="seconds of suspend that trigger a restart")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--settle", type=float, default=15.0, help="wait after a resume before restarting (GPU settle)")
    ap.add_argument("--max-restarts", type=int, default=100)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command ...")
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command given (use: suspend_resilient.py -- <command> ...)")

    state = {"child": None, "simulate": False, "quit": None}

    def forward(signum, _frame):
        state["quit"] = signum
        if state["child"] is not None:
            stop_group(state["child"], signum, grace=60.0)

    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, forward)
    signal.signal(signal.SIGUSR1, lambda *_: state.__setitem__("simulate", True))

    restarts = 0
    while True:
        # own process group: the whole tree (pipeline -> ns-train -> ...) can be killed at once
        p = subprocess.Popen(cmd, start_new_session=True)
        state["child"] = p
        log(f"started pid {p.pid}" + (f" (restart {restarts})" if restarts else "") + f": {' '.join(cmd)[:160]}")
        base, resumed = suspended_seconds(), None
        while p.poll() is None:
            time.sleep(a.poll)
            if state["quit"] is not None:
                break
            gap = suspended_seconds() - base
            if gap > a.min_suspend or state["simulate"]:
                resumed = gap
                break
        if state["quit"] is not None:
            p.wait()
            log(f"stopped by signal {state['quit']}")
            return 128 + int(state["quit"])
        if resumed is None:
            log(f"command finished with exit code {p.returncode}")
            return p.returncode
        state["simulate"] = False
        log(f"system was suspended for {resumed:.0f} s; CUDA in the running command does not survive a suspend "
            f"-> stopping it and resuming from the last checkpoint")
        stop_group(p)
        restarts += 1
        if restarts > a.max_restarts:
            log("too many restarts; giving up")
            return 1
        time.sleep(a.settle)


if __name__ == "__main__":
    sys.exit(main())
