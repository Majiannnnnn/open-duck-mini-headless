"""Read the classroom release's discrete reference gait."""

from dataclasses import dataclass
from itertools import product
from math import prod
from operator import index
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


CONTACTS = slice(32, 34)  # Left, right. Values above 0.5 indicate stance.
_ARCHIVE_FIELDS = {
    "frames", "roots", "commands", "source_commands", "home", "period_steps", "dt",
}


@dataclass(frozen=True)
class ReferenceMotion:
    commands: NDArray[np.float32]
    frames: NDArray[np.float32]
    home: NDArray[np.float64]
    dt: float

    def frame(self, command: ArrayLike, phase_index: int) -> NDArray[np.float32]:
        """Select the nearest command on each axis and wrap the phase index."""
        command = np.asarray(command, dtype=self.commands.dtype)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Reference command must be three finite values: vx, vy, yaw_rate")
        delta = np.abs(self.commands - command)
        matching = np.all(delta == delta.min(axis=0), axis=1)
        command_index = np.flatnonzero(matching)[0]  # The sorted grid breaks ties low.
        phase = index(phase_index) % self.frames.shape[1]
        return self.frames[command_index, phase]


def _check_array(name: str, array: np.ndarray, shape: tuple[int, ...], dtype: type) -> None:
    if array.shape != shape or array.dtype != dtype or not np.isfinite(array).all():
        raise ValueError(f"Invalid reference array {name}: expected finite {dtype.__name__} {shape}")


def load_reference(path: Path, *, period_steps: int) -> ReferenceMotion:
    """Read and validate the complete 180-command, 40-channel gait archive."""
    if index(period_steps) <= 0:
        raise ValueError("Reference period must contain at least one control step")

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != _ARCHIVE_FIELDS:
            raise ValueError(f"Invalid reference archive fields in {path}")
        frames = archive["frames"]
        commands = archive["commands"]
        home = archive["home"]
        stored_period = archive["period_steps"]
        dt = archive["dt"]
        _check_array("frames", frames, (180, period_steps, 40), np.float32)
        _check_array("commands", commands, (180, 3), np.float32)
        _check_array("home", home, (17,), np.float64)
        _check_array("roots", archive["roots"], (180, period_steps, 3), np.float32)
        _check_array("source_commands", archive["source_commands"], (180, 3), np.float32)
        _check_array("period_steps", stored_period, (), np.int64)
        _check_array("dt", dt, (), np.float64)

    if stored_period.item() != period_steps or dt.item() <= 0.0:
        raise ValueError("Reference clock metadata does not match the configured period")
    axes = [np.unique(commands[:, axis]) for axis in range(3)]
    if prod(map(len, axes)) != len(commands):
        raise ValueError("Reference commands must form a complete, sorted Cartesian grid")
    expected_grid = np.asarray(list(product(*axes)), dtype=commands.dtype)
    if not np.array_equal(commands, expected_grid):
        raise ValueError("Reference commands must form a complete, sorted Cartesian grid")

    return ReferenceMotion(commands, frames, home, dt.item())
