"""Parent-side handle on an episode subprocess."""

from __future__ import annotations

import multiprocessing as mp

from .config import EnvConfig
from .spec import ScenarioSpec

#: Generous — a full 7-year horizon is ~26 s, and the final `step` runs the whole
#: remaining tail of the simulation in one go.
DEFAULT_TIMEOUT = 900.0


class EpisodeProcessError(RuntimeError):
    """The episode subprocess failed. The child's traceback is included verbatim."""


class EpisodeProcess:
    """Runs one episode in a fresh interpreter, driven over a pipe."""

    def __init__(
        self,
        config: EnvConfig,
        spec: ScenarioSpec,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.config = config
        self.spec = spec
        self.timeout = timeout
        self._proc: mp.process.BaseProcess | None = None
        self._conn = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    def start(self, seed: int | None = None):
        """Spawn the child and run to the first decision boundary."""
        from .worker import episode_worker

        self.close()
        # "spawn" rather than "fork": a forked child would inherit the parent's already
        # imported RuFaS modules and their singleton state, which is exactly the
        # cross-episode contamination the subprocess model exists to avoid.
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=episode_worker,
            args=(child_conn, self.config, self.spec),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()  # the parent must not hold the child's end open
        self._conn = parent_conn
        return self._request("start", seed)

    def step(self, action):
        return self._request("step", action)

    def close(self) -> None:
        """Shut the child down, escalating if it does not go quietly."""
        if self._proc is None:
            return
        try:
            if self._conn is not None and self._proc.is_alive():
                try:
                    self._conn.send(("close", None))
                    if self._conn.poll(30.0):
                        self._conn.recv()
                except (BrokenPipeError, EOFError, OSError):
                    pass
            self._proc.join(timeout=30.0)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=10.0)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=10.0)
        finally:
            if self._conn is not None:
                try:
                    self._conn.close()
                except OSError:
                    pass
            self._conn = None
            self._proc = None

    def __enter__(self) -> "EpisodeProcess":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- internals -------------------------------------------------------------

    def _request(self, command: str, payload):
        if self._conn is None or self._proc is None:
            raise EpisodeProcessError("Episode process is not running; call start() first")

        try:
            self._conn.send((command, payload))
        except (BrokenPipeError, OSError) as exc:
            raise self._died(f"Episode process died before it could receive {command!r}") from exc

        if not self._conn.poll(self.timeout):
            self.close()
            raise EpisodeProcessError(
                f"Episode process did not respond to {command!r} within {self.timeout:.0f}s. "
                "A pause hook that never fires, or a simulation stuck between boundaries, "
                "both look like this."
            )

        try:
            status, result = self._conn.recv()
        except (EOFError, ConnectionResetError) as exc:
            raise self._died(f"Episode process died while handling {command!r}") from exc

        if status == "error":
            self.close()
            raise EpisodeProcessError(f"Episode subprocess failed:\n{result}")
        return result

    def _died(self, message: str) -> EpisodeProcessError:
        """Build the error for a dead child, tearing the process down on the way out."""
        exitcode = self._proc.exitcode if self._proc is not None else None
        self.close()
        return EpisodeProcessError(f"{message} (exit code {exitcode})")
