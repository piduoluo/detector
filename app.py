from __future__ import annotations

import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
SMOKING_MODEL_PATH = BASE_DIR / "smoking.pt"
MASK_MODEL_PATH = BASE_DIR / "trained_mask_model.pt"

SMOKING_CONF_THRESHOLD = 0.35
MASK_CONF_THRESHOLD = 0.35

# Smarter screenshot policy: reduce false positives + avoid spamming disk.
SAVE_STREAK_THRESHOLD = 3
SAVE_COOLDOWN_SECONDS = 10.0
# If a single continuous violation lasts a long time, re-save evidence periodically.
LONG_EVENT_RESAVE_SECONDS = 60.0
MAX_CAPTURE_FILES = 200
CAPTURE_DIR = BASE_DIR / "captures"
CAPTURE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
smoking_model = YOLO(str(SMOKING_MODEL_PATH))
mask_model = YOLO(str(MASK_MODEL_PATH)) if MASK_MODEL_PATH.exists() else None
screenshot_saved = False
screenshot_path = ""
violation_streak = 0
last_violation_key = ""
last_save_time = 0.0
event_active = False
event_saved = False
event_start_time = 0.0
event_last_saved_time = 0.0

def _norm_class_name(name: object) -> str:
    value = str(name).strip().lower()
    return value.replace("_", "-").replace(" ", "-")


def _is_smoking_class(class_id: int, class_name: str) -> bool:
    normalized = _norm_class_name(class_name)
    return normalized in {"cigarette", "smoking"} or class_id == 0


def _is_no_mask_class(class_id: int, class_name: str) -> bool:
    normalized = _norm_class_name(class_name)
    # Trained model uses "no-mask"; accept common variants to be robust.
    if normalized in {"no-mask", "nomask", "no-mask-face", "without-mask", "mask-off"}:
        return True
    return class_id == 0 and "mask" in normalized and "no" in normalized


def _decode_image_from_request() -> np.ndarray:
    if "image" in request.files:
        image_file = request.files["image"]
        pil_image = Image.open(image_file.stream).convert("RGB")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    data_url = request.form.get("image_data", "").strip()
    if data_url.startswith("data:image"):
        _, encoded = data_url.split(",", 1)
        raw_bytes = base64.b64decode(encoded)
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    raise ValueError("No valid image found in request.")


def _encode_image_to_data_url(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def _violation_key(smoking_detected: bool, no_mask_detected: bool) -> str:
    parts: list[str] = []
    if smoking_detected:
        parts.append("smoking")
    if no_mask_detected:
        parts.append("no-mask")
    return "+".join(parts) if parts else "none"


def _rotate_captures() -> None:
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    files = sorted(CAPTURE_DIR.glob("violation_*.jpg"), key=_mtime, reverse=True)
    for old in files[MAX_CAPTURE_FILES:]:
        try:
            old.unlink()
        except OSError:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    global screenshot_saved, screenshot_path, violation_streak, last_violation_key, last_save_time
    global event_active, event_saved, event_start_time, event_last_saved_time

    try:
        frame = _decode_image_from_request()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    smoking_results = smoking_model.predict(
        source=frame, conf=SMOKING_CONF_THRESHOLD, verbose=False
    )
    smoking_result = smoking_results[0]

    smoking_names = smoking_model.names
    smoking_boxes = []
    for box in smoking_result.boxes:
        class_id = int(box.cls[0])
        class_name = str(smoking_names.get(class_id, class_id))
        confidence = float(box.conf[0])
        if _is_smoking_class(class_id, class_name):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            smoking_boxes.append(
                {
                    "class_id": class_id,
                    "class_name": _norm_class_name(class_name),
                    "confidence": round(confidence, 4),
                    "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )

    no_mask_boxes = []
    if mask_model is not None:
        mask_results = mask_model.predict(
            source=frame, conf=MASK_CONF_THRESHOLD, verbose=False
        )
        mask_result = mask_results[0]
        mask_names = mask_model.names
        for box in mask_result.boxes:
            class_id = int(box.cls[0])
            class_name = str(mask_names.get(class_id, class_id))
            confidence = float(box.conf[0])
            if _is_no_mask_class(class_id, class_name):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                no_mask_boxes.append(
                    {
                        "class_id": class_id,
                        "class_name": _norm_class_name(class_name),
                        "confidence": round(confidence, 4),
                        "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    }
                )

    annotated = frame.copy()
    for entry in smoking_boxes:
        x1, y1, x2, y2 = map(int, entry["xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            annotated,
            f"smoking {entry['confidence']:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    for entry in no_mask_boxes:
        x1, y1, x2, y2 = map(int, entry["xyxy"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(
            annotated,
            f"no-mask {entry['confidence']:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    preview_url = _encode_image_to_data_url(annotated)
    smoking_detected = len(smoking_boxes) > 0
    no_mask_detected = len(no_mask_boxes) > 0
    violation_detected = smoking_detected or no_mask_detected
    violation_key = _violation_key(smoking_detected, no_mask_detected)
    just_saved = False
    save_reason = ""
    save_streak = 0

    now = time.time()
    if violation_detected:
        if violation_key == last_violation_key:
            violation_streak += 1
        else:
            violation_streak = 1
            last_violation_key = violation_key
            # New violation type during a continuous run: treat as new event.
            event_saved = False

        if violation_streak == 1:
            event_active = True
            event_start_time = now
    else:
        violation_streak = 0
        last_violation_key = ""
        event_active = False
        event_saved = False
        event_start_time = 0.0

    can_save_globally = (now - last_save_time) >= SAVE_COOLDOWN_SECONDS
    is_long_event_resave = (
        event_active
        and event_saved
        and LONG_EVENT_RESAVE_SECONDS > 0
        and (now - event_last_saved_time) >= LONG_EVENT_RESAVE_SECONDS
    )
    should_save = (
        violation_detected
        and violation_streak >= SAVE_STREAK_THRESHOLD
        and can_save_globally
        and (not event_saved or is_long_event_resave)
    )

    if should_save:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        save_name = f"violation_{timestamp}_{violation_key}.jpg"
        full_path = CAPTURE_DIR / save_name
        cv2.imwrite(str(full_path), annotated)
        _rotate_captures()

        screenshot_saved = True
        screenshot_path = str(full_path)
        last_save_time = now
        just_saved = True
        save_reason = violation_key
        save_streak = violation_streak
        event_saved = True
        event_last_saved_time = now

    return jsonify(
        {
            "ok": True,
            "mask_model_loaded": mask_model is not None,
            "violation_detected": violation_detected,
            "violation_key": violation_key,
            "violation_streak": violation_streak,
            "smoking_detected": smoking_detected,
            "no_mask_detected": no_mask_detected,
            "smoking_count": len(smoking_boxes),
            "no_mask_count": len(no_mask_boxes),
            "boxes": smoking_boxes,
            "no_mask_boxes": no_mask_boxes,
            "preview_image": preview_url,
            "screenshot_saved": screenshot_saved,
            "just_saved": just_saved,
            "screenshot_path": screenshot_path,
            "save_reason": save_reason,
            "save_streak": save_streak,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
