from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from .config import ModelConfig


def _norm_class_name(name: object) -> str:
    value = str(name).strip().lower()
    return value.replace("_", "-").replace(" ", "-")


def _is_smoking_class(class_id: int, class_name: str) -> bool:
    normalized = _norm_class_name(class_name)
    return normalized in {"cigarette", "smoking"} or class_id == 0


def _is_no_mask_class(class_id: int, class_name: str) -> bool:
    normalized = _norm_class_name(class_name)
    if normalized in {"no-mask", "nomask", "no-mask-face", "without-mask", "mask-off"}:
        return True
    return class_id == 0 and "mask" in normalized and "no" in normalized


@dataclass(frozen=True)
class DetectedBox:
    class_id: int
    class_name: str
    confidence: float
    xyxy: list[float]


@dataclass(frozen=True)
class DetectionResult:
    smoking_boxes: list[DetectedBox]
    no_mask_boxes: list[DetectedBox]
    smoking_detected: bool
    no_mask_detected: bool
    violation_detected: bool
    violation_key: str


class YOLODetector:
    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._smoking_model = YOLO(str(config.smoking_model_path))
        self._mask_model = (
            YOLO(str(config.mask_model_path)) if config.mask_model_path.exists() else None
        )

    @property
    def mask_model_loaded(self) -> bool:
        return self._mask_model is not None

    @staticmethod
    def _violation_key(*, smoking_detected: bool, no_mask_detected: bool) -> str:
        parts: list[str] = []
        if smoking_detected:
            parts.append("smoking")
        if no_mask_detected:
            parts.append("no-mask")
        return "+".join(parts) if parts else "none"

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        smoking_results = self._smoking_model.predict(
            source=frame_bgr,
            conf=self._config.smoking_conf_threshold,
            verbose=False,
        )
        smoking_result = smoking_results[0]

        smoking_names = self._smoking_model.names
        smoking_boxes: list[DetectedBox] = []
        for box in smoking_result.boxes:
            class_id = int(box.cls[0])
            class_name = str(smoking_names.get(class_id, class_id))
            confidence = float(box.conf[0])
            if _is_smoking_class(class_id, class_name):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                smoking_boxes.append(
                    DetectedBox(
                        class_id=class_id,
                        class_name=_norm_class_name(class_name),
                        confidence=round(confidence, 4),
                        xyxy=[round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    )
                )

        no_mask_boxes: list[DetectedBox] = []
        if self._mask_model is not None:
            mask_results = self._mask_model.predict(
                source=frame_bgr,
                conf=self._config.mask_conf_threshold,
                verbose=False,
            )
            mask_result = mask_results[0]
            mask_names = self._mask_model.names
            for box in mask_result.boxes:
                class_id = int(box.cls[0])
                class_name = str(mask_names.get(class_id, class_id))
                confidence = float(box.conf[0])
                if _is_no_mask_class(class_id, class_name):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    no_mask_boxes.append(
                        DetectedBox(
                            class_id=class_id,
                            class_name=_norm_class_name(class_name),
                            confidence=round(confidence, 4),
                            xyxy=[
                                round(x1, 1),
                                round(y1, 1),
                                round(x2, 1),
                                round(y2, 1),
                            ],
                        )
                    )

        smoking_detected = len(smoking_boxes) > 0
        no_mask_detected = len(no_mask_boxes) > 0
        violation_detected = smoking_detected or no_mask_detected
        violation_key = self._violation_key(
            smoking_detected=smoking_detected, no_mask_detected=no_mask_detected
        )

        return DetectionResult(
            smoking_boxes=smoking_boxes,
            no_mask_boxes=no_mask_boxes,
            smoking_detected=smoking_detected,
            no_mask_detected=no_mask_detected,
            violation_detected=violation_detected,
            violation_key=violation_key,
        )

