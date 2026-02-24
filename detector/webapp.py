from __future__ import annotations

import cv2
from flask import Flask, jsonify, render_template, request

from .capture_policy import CapturePolicy
from .config import AppConfig
from .image_io import decode_image_from_request, encode_image_to_data_url
from .yolo_detector import YOLODetector


def create_app(config: AppConfig) -> Flask:
    app = Flask(__name__)
    detector = YOLODetector(config.models)
    capture_policy = CapturePolicy(config.capture_dir, config.capture)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/detect", methods=["POST"])
    def detect():
        try:
            frame = decode_image_from_request(request)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

        detection = detector.detect(frame)

        annotated = frame.copy()
        for entry in detection.smoking_boxes:
            x1, y1, x2, y2 = map(int, entry.xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                f"smoking {entry.confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        for entry in detection.no_mask_boxes:
            x1, y1, x2, y2 = map(int, entry.xyxy)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated,
                f"no-mask {entry.confidence:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        preview_url = encode_image_to_data_url(annotated)
        save = capture_policy.on_frame(
            violation_detected=detection.violation_detected,
            violation_key=detection.violation_key,
            annotated_bgr=annotated,
        )

        return jsonify(
            {
                "ok": True,
                "mask_model_loaded": detector.mask_model_loaded,
                "violation_detected": detection.violation_detected,
                "violation_key": detection.violation_key,
                "violation_streak": capture_policy.violation_streak,
                "smoking_detected": detection.smoking_detected,
                "no_mask_detected": detection.no_mask_detected,
                "smoking_count": len(detection.smoking_boxes),
                "no_mask_count": len(detection.no_mask_boxes),
                "boxes": [box.__dict__ for box in detection.smoking_boxes],
                "no_mask_boxes": [box.__dict__ for box in detection.no_mask_boxes],
                "preview_image": preview_url,
                "screenshot_saved": save.screenshot_saved,
                "just_saved": save.just_saved,
                "screenshot_path": save.screenshot_path,
                "save_reason": save.save_reason,
                "save_streak": save.save_streak,
            }
        )

    return app

