from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CaptureConfig


@dataclass
class SaveDecision:
    screenshot_saved: bool
    just_saved: bool
    screenshot_path: str
    save_reason: str
    save_streak: int


@dataclass
class _CaptureState:
    violation_streak: int = 0
    last_violation_key: str = ""
    last_save_time: float = 0.0
    event_active: bool = False
    event_saved: bool = False
    event_start_time: float = 0.0
    event_last_saved_time: float = 0.0


class CapturePolicy:
    def __init__(self, capture_dir: Path, config: CaptureConfig) -> None:
        self._capture_dir = capture_dir
        self._config = config
        self._state = _CaptureState()
        self._capture_dir.mkdir(exist_ok=True)

    @property
    def violation_streak(self) -> int:
        return self._state.violation_streak

    @staticmethod
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _rotate_captures(self) -> None:
        files = sorted(
            self._capture_dir.glob("violation_*.jpg"),
            key=self._mtime,
            reverse=True,
        )
        for old in files[self._config.max_capture_files :]:
            try:
                old.unlink()
            except OSError:
                pass

    def on_frame(
        self,
        *,
        violation_detected: bool,
        violation_key: str,
        annotated_bgr: np.ndarray,
        now: float | None = None,
    ) -> SaveDecision:
        now = time.time() if now is None else now
        state = self._state

        if violation_detected:
            if violation_key == state.last_violation_key:
                state.violation_streak += 1
            else:
                state.violation_streak = 1
                state.last_violation_key = violation_key
                state.event_saved = False

            if state.violation_streak == 1:
                state.event_active = True
                state.event_start_time = now
        else:
            state.violation_streak = 0
            state.last_violation_key = ""
            state.event_active = False
            state.event_saved = False
            state.event_start_time = 0.0

        can_save_globally = (now - state.last_save_time) >= self._config.save_cooldown_seconds
        is_long_event_resave = (
            state.event_active
            and state.event_saved
            and self._config.long_event_resave_seconds > 0
            and (now - state.event_last_saved_time) >= self._config.long_event_resave_seconds
        )
        should_save = (
            violation_detected
            and state.violation_streak >= self._config.save_streak_threshold
            and can_save_globally
            and (not state.event_saved or is_long_event_resave)
        )

        if not should_save:
            return SaveDecision(
                screenshot_saved=False,
                just_saved=False,
                screenshot_path="",
                save_reason="",
                save_streak=0,
            )

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        save_name = f"violation_{timestamp}_{violation_key}.jpg"
        full_path = self._capture_dir / save_name
        cv2.imwrite(str(full_path), annotated_bgr)
        self._rotate_captures()

        state.last_save_time = now
        state.event_saved = True
        state.event_last_saved_time = now

        return SaveDecision(
            screenshot_saved=True,
            just_saved=True,
            screenshot_path=str(full_path),
            save_reason=violation_key,
            save_streak=state.violation_streak,
        )

