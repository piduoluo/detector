import cv2
import time
import os
from ultralytics import YOLO

# --- 1. 配置区域 ---
# 吸烟模型 + 口罩模型
SMOKING_MODEL_PATH = "smoking.pt"
MASK_MODEL_PATH = "trained_mask_model.pt"

# 报警冷却时间
ALERT_COOLDOWN = 3.0 
last_alert_time = 0

# --- 2. 加载模型 ---
print(f"🛡️ 正在加载吸烟模型: {SMOKING_MODEL_PATH}...")
try:
    smoking_model = YOLO(SMOKING_MODEL_PATH)
    print("✅ 吸烟模型加载成功。")
    print(f"吸烟模型类别: {smoking_model.names}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

mask_model = None
if os.path.exists(MASK_MODEL_PATH):
    print(f"🛡️ 正在加载口罩模型: {MASK_MODEL_PATH}...")
    try:
        mask_model = YOLO(MASK_MODEL_PATH)
        print("✅ 口罩模型加载成功。")
        print(f"口罩模型类别: {mask_model.names}")
    except Exception as e:
        print(f"⚠️ 口罩模型加载失败，将仅启用吸烟检测: {e}")

# --- 3. 报警函数 ---
def trigger_alert(message):
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time > ALERT_COOLDOWN:
        print(f"🚨 {message}")
        os.system(f"say {message} &") 
        last_alert_time = current_time

def _norm(name):
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def _is_smoking(class_id, class_name):
    n = _norm(class_name)
    return n in {"cigarette", "smoking"} or class_id == 0


def _is_no_mask(class_id, class_name):
    n = _norm(class_name)
    if n in {"no-mask", "nomask", "without-mask", "mask-off"}:
        return True
    return class_id == 0 and "mask" in n and "no" in n

# --- 4. 启动摄像头 ---
cap = cv2.VideoCapture(0)

# --- 5. 核心循环 ---
while True:
    ret, frame = cap.read()
    if not ret: break

    # 推理
    smoking_results = smoking_model(frame, stream=True, conf=0.4)
    mask_results = mask_model(frame, stream=True, conf=0.4) if mask_model else None
    smoking_detected = False
    no_mask_detected = False
    
    for result in smoking_results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            cls_name = smoking_model.names.get(cls_id, cls_id)
            conf = float(box.conf[0])
            if not _is_smoking(cls_id, cls_name):
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"smoking {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            smoking_detected = True

    if mask_results:
        for result in mask_results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = mask_model.names.get(cls_id, cls_id)
                conf = float(box.conf[0])
                if not _is_no_mask(cls_id, cls_name):
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = f"no-mask {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                no_mask_detected = True

    if smoking_detected or no_mask_detected:
        msg = []
        if smoking_detected:
            msg.append("发现吸烟行为")
        if no_mask_detected:
            msg.append("发现未佩戴口罩")
        trigger_alert("，".join(msg))

    # 视觉报警特效
    if smoking_detected or no_mask_detected:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)

    cv2.imshow('Kitchen Sentinel (Real Model)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
