from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaptureConfig:
    save_streak_threshold: int = 3
    save_cooldown_seconds: float = 10.0
    long_event_resave_seconds: float = 60.0
    max_capture_files: int = 200


@dataclass(frozen=True)
class ModelConfig:
    smoking_model_path: Path
    mask_model_path: Path
    smoking_conf_threshold: float = 0.35
    mask_conf_threshold: float = 0.35


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    capture_dir: Path
    models: ModelConfig
    capture: CaptureConfig = CaptureConfig()
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = True


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_config(base_dir: Path) -> AppConfig:
    smoking_model_path = Path(os.getenv("SMOKING_MODEL_PATH", base_dir / "smoking.pt"))
    mask_model_path = Path(
        os.getenv("MASK_MODEL_PATH", base_dir / "trained_mask_model.pt")
    )
    capture_dir = Path(os.getenv("CAPTURE_DIR", base_dir / "captures"))

    models = ModelConfig(
        smoking_model_path=smoking_model_path,
        mask_model_path=mask_model_path,
        smoking_conf_threshold=_env_float("SMOKING_CONF_THRESHOLD", 0.35),
        mask_conf_threshold=_env_float("MASK_CONF_THRESHOLD", 0.35),
    )

    capture = CaptureConfig(
        save_streak_threshold=_env_int("SAVE_STREAK_THRESHOLD", 3),
        save_cooldown_seconds=_env_float("SAVE_COOLDOWN_SECONDS", 10.0),
        long_event_resave_seconds=_env_float("LONG_EVENT_RESAVE_SECONDS", 60.0),
        max_capture_files=_env_int("MAX_CAPTURE_FILES", 200),
    )

    return AppConfig(
        base_dir=base_dir,
        capture_dir=capture_dir,
        models=models,
        capture=capture,
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 5000),
        debug=os.getenv("DEBUG", "true").strip().lower() in {"1", "true", "yes", "y"},
    )

