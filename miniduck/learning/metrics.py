"""Persistence for PPO progress, metrics, and verified checkpoints."""

from copy import deepcopy
import json
from math import isfinite
from operator import index
from os import fsync
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from time import monotonic

import numpy as np


def _step_count(value) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Environment steps must be a nonnegative integer")
    step = index(value)
    if step < 0:
        raise ValueError("Environment steps must be a nonnegative integer")
    return step


def _encode(value: dict) -> str:
    return json.dumps(value, indent=2, allow_nan=False) + "\n"


def _atomic_write(path: Path, contents: str) -> None:
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(contents)
            stream.flush()
            fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _metric_value(name: str, value):
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        raise ValueError(f"Training metric {name!r} must be numeric")
    if array.dtype.kind == "f":
        array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"Nonfinite training metric {name!r}")
    return array.tolist()


class RunLog:
    """Own one newly created run directory and its training records."""

    def __init__(self, output: Path | str, run_config: dict, policy_contract: dict):
        config_json = _encode(run_config)
        contract_json = _encode(policy_contract)
        self._budget = _step_count(run_config["ppo"]["num_timesteps"])
        if self._budget == 0:
            raise ValueError("Training budget must be positive")

        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        self.output = output.resolve()
        self._lock = Lock()
        self._started_at = monotonic()
        self._progress = {
            "status": "training",
            "complete": False,
            "budget_complete": False,
            "planned_actor_samples": self._budget,
            "actual_new_actor_samples": 0,
            "checkpoints": [],
        }
        self._write_progress(self._progress)
        try:
            _atomic_write(self.output / "run_config.json", config_json)
            _atomic_write(self.output / "policy_contract.json", contract_json)
            (self.output / "metrics.jsonl").touch(exist_ok=False)
        except Exception as error:
            self.fail(error)
            raise

    @property
    def progress(self) -> dict:
        with self._lock:
            return deepcopy(self._progress)

    def _ensure_running(self) -> None:
        if self._progress["status"] != "training":
            raise RuntimeError("The training run has already finished")

    def _write_progress(self, progress: dict) -> None:
        _atomic_write(self.output / "progress.json", _encode(progress))
        self._progress = progress

    def record_metrics(self, step: int, metrics: dict) -> None:
        """Callback for training and evaluation metrics at actor sample steps."""
        step = _step_count(step)
        reserved = {"steps", "elapsed_seconds"}
        if not all(isinstance(name, str) and name not in reserved for name in metrics):
            raise ValueError("Training metric names must be strings distinct from the log metadata")
        values = {name: _metric_value(name, value) for name, value in metrics.items()}
        with self._lock:
            self._ensure_running()
            row = {"steps": step, "elapsed_seconds": monotonic() - self._started_at, **values}
            with (self.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if step > self._progress["actual_new_actor_samples"]:
                self._write_progress({**self._progress, "actual_new_actor_samples": step})

    def record_checkpoint(self, path: Path | str, step: int, verification_error: float) -> None:
        """Publish a saved checkpoint only after its reload check passes."""
        step = _step_count(step)
        verification_error = float(verification_error)
        if not isfinite(verification_error) or verification_error < 0:
            raise ValueError("Checkpoint verification error must be finite and nonnegative")
        path = Path(path).resolve(strict=True)
        relative_path = path.relative_to(self.output)
        if not path.is_dir() or relative_path == Path("."):
            raise ValueError("Checkpoint must be a directory inside the run")
        with self._lock:
            self._ensure_running()
            checkpoints = self._progress["checkpoints"]
            if any(entry["step"] == step for entry in checkpoints):
                raise ValueError(f"A checkpoint was already recorded at step {step}")
            entry = {
                "step": step,
                "path": relative_path.as_posix(),
                "verification_max_abs_error": verification_error,
            }
            self._write_progress({
                **self._progress,
                "checkpoints": [*checkpoints, entry],
                "actual_new_actor_samples": max(step, self._progress["actual_new_actor_samples"]),
            })

    def complete(self, step: int) -> None:
        """Mark the planned budget and final checkpoint as completed."""
        step = _step_count(step)
        with self._lock:
            self._ensure_running()
            if step != self._budget or self._progress["actual_new_actor_samples"] != step:
                raise ValueError(f"Expected {self._budget} actor samples, received {step}")
            if not any(entry["step"] == step for entry in self._progress["checkpoints"]):
                raise ValueError("The final verified checkpoint has not been recorded")
            self._write_progress({
                **self._progress,
                "status": "budget_complete_unvalidated",
                "complete": True,
                "budget_complete": True,
            })

    def fail(self, error: BaseException) -> None:
        """Preserve all recorded samples and checkpoints with the failure."""
        with self._lock:
            if self._progress["status"] == "training":
                self._write_progress({
                    **self._progress,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                })
