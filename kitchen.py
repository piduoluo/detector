import cv2
import time
import os
from ultralytics import YOLO

# --- 1. 配置区域 ---
# 换上我们刚才训练好的“真模型”
model_path = 'smoking.pt' 

# 报警冷却时间
ALERT_COOLDOWN = 3.0 
last_alert_time = 0

# --- 2. 加载模型 ---
print(f"🛡️ 正在加载你的专属模型: {model_path}...")
try:
    model = YOLO(model_path)
    print("✅ 模型加载成功！它现在是一名'香烟捕手'了。")
    # 打印一下模型认识的所有类别，确认里面有 cigarette
    print(f"模型识别类别: {model.names}")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()

# --- 3. 报警函数 ---
def trigger_alert(message):
    global last_alert_time
    current_time = time.time()
    if current_time - last_alert_time > ALERT_COOLDOWN:
        print(f"🚨 {message}")
        os.system(f"say {message} &") 
        last_alert_time = current_time

# --- 4. 启动摄像头 ---
cap = cv2.VideoCapture(0)

# --- 5. 核心循环 ---
while True:
    ret, frame = cap.read()
    if not ret: break

    # 推理
    # conf=0.4: 因为我们是极速训练，模型可能稍微有点不自信，把门槛设低一点点
    results = model(frame, stream=True, conf=0.4)

    violation_detected = False
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # 获取类别名称
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            
            # 获取置信度 (0.0 - 1.0)
            conf = float(box.conf[0])

            # --- ⚠️ 真正的实战判断 ---
            # 因为这个模型是用香烟数据专门微调的，
            # 所以不管它返回 ID 是 0 还是别的，只要名字里包含 cigarette 或者是类别 0
            # 我们都认为它是香烟
            
            # 画框框
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 只有检测到香烟时才报警
            # (通常训练后的模型，类别名字就是 'cigarette')
            label = f"{cls_name} {conf:.2f}"
            
            # 绘制红框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            trigger_alert("发现吸烟行为")
            violation_detected = True

    # 视觉报警特效
    if violation_detected:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)

    cv2.imshow('Kitchen Sentinel (Real Model)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()