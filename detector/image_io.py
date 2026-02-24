from __future__ import annotations

import base64
import io

import cv2
import numpy as np
from flask import Request
from PIL import Image


def decode_image_from_request(request: Request) -> np.ndarray:
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


def encode_image_to_data_url(image_bgr: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image.")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

