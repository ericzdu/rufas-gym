"""The episode subprocess.

Each episode gets its own interpreter. This is not tidiness — a second RuFaS run in the
same process is roughly 5x slower than the first while producing identical output, so
reusing an interpreter across episodes would make training progressively slower for no
reason. RuFaS's own task manager sidesteps the same thing with `maxtasksperchild=1`.

The child owns the `Episode` (and therefore the paused worker thread); the parent holds
only a pipe. The protocol is strictly synchronous request/response:

    parent -> child   ("start", seed) | ("step", action) | ("close", None)
    child  -> parent  ("ok", payload) | ("error", traceback_text)

Once an error is reported the child stops; the parent tears the process down and raises.
"""

from __future__ import annotations

import os
import sys
import traceback

from .bootstrap import rufas_cwd
from .config import EnvConfig
from .spec import ScenarioSpec


def _silence_stdout() -> None:
    """Send RuFaS's progress chatter to /dev/null.

    RuFaS prints task banners on every run; across thousands of training episodes that is
    pure noise. Set RUFAS_RL_VERBOSE=1 to keep it. stderr is left alone so real failures
    stay visible.
    """
    if os.environ.get("RUFAS_RL_VERBOSE"):
        return
    sys.stdout.flush()
    devnull = open(os.devnull, "w")
    sys.stdout = devnull


def episode_worker(conn, config: EnvConfig, spec: ScenarioSpec) -> None:
    """Child-process entry point. Serves one episode until told to close."""
    from .episode import Episode

    _silence_stdout()
    episode = None
    try:
        with rufas_cwd():
            episode = Episode(config, spec)
            while True:
                try:
                    command, payload = conn.recv()
                except EOFError:
                    break  # parent went away

                if command == "close":
                    conn.send(("ok", None))
                    break

                try:
                    if command == "start":
                        conn.send(("ok", episode.start(seed=payload)))
                    elif command == "step":
                        conn.send(("ok", episode.step(payload)))
                    else:
                        raise ValueError(f"Unknown command {command!r}")
                except BaseException:  # noqa: BLE001 — report, then stop serving
                    conn.send(("error", traceback.format_exc()))
                    break
    except BaseException:  # noqa: BLE001 — failure during setup, before the loop
        try:
            conn.send(("error", traceback.format_exc()))
        except BaseException:  # noqa: BLE001 — pipe already broken
            pass
    finally:
        if episode is not None:
            episode.close()
        try:
            conn.close()
        except BaseException:  # noqa: BLE001
            pass
