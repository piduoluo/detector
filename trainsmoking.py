from ultralytics import YOLO

model = YOLO('yolov8n.pt')

print("🚀 启动优化版训练 (Mac M4 特调)...")

# 核心优化点：
# 1. imgsz=416: 平衡点。比320清楚，比640快很多。
# 2. workers=0: 关键！Mac上设为0（单线程加载）往往能解决卡顿问题。
# 3. batch=8: 降低显存压力。

try:
    results = model.train(
        data='smoking.yaml',
        epochs=30,         # 30轮足够验证了
        imgsz=416,         # <--- 黄金尺寸
        batch=8,           # <--- 减小批次
        workers=0,         # <--- 关键！解决 Mac 读取卡死
        device='mps',
        name='smoking_optimized'
    )
    print("✅ 训练完成！")
except KeyboardInterrupt:
    print("停止")