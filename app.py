from __future__ import annotations

import base64
import io
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "smoking.pt"
CONF_THRESHOLD = 0.35
CAPTURE_DIR = BASE_DIR / "captures"
CAPTURE_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
model = YOLO(str(MODEL_PATH))
screenshot_saved = False
screenshot_path = ""


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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    global screenshot_saved, screenshot_path

    try:
        frame = _decode_image_from_request()
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    names = model.names
    smoking_boxes = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = str(names.get(class_id, class_id)).lower()
        confidence = float(box.conf[0])
        if class_name == "cigarette" or class_id == 0:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            smoking_boxes.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                }
            )

    annotated = result.plot()
    preview_url = _encode_image_to_data_url(annotated)
    smoking_detected = len(smoking_boxes) > 0
    just_saved = False

    if smoking_detected and not screenshot_saved:
        save_name = "smoking_detected_once.jpg"
        full_path = CAPTURE_DIR / save_name
        cv2.imwrite(str(full_path), annotated)
        screenshot_saved = True
        screenshot_path = str(full_path)
        just_saved = True

    return jsonify(
        {
            "ok": True,
            "smoking_detected": smoking_detected,
            "count": len(smoking_boxes),
            "boxes": smoking_boxes,
            "preview_image": preview_url,
            "screenshot_saved": screenshot_saved,
            "just_saved": just_saved,
            "screenshot_path": screenshot_path,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
