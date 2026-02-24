# 项目导读（给人和 AI）

## 这个项目在做什么
浏览器端采集摄像头帧 → 以图片形式 `POST /api/detect` → 后端用 YOLO 做两类违规检测：
- 吸烟（模型：`smoking.pt`）
- 未戴口罩（模型：`trained_mask_model.pt`，可选）

后端会返回“标注后的预览图”（DataURL），并在满足策略时把违规证据截图保存到 `captures/`。

## 关键入口
- 运行：`python app.py`
- 页面：`http://127.0.0.1:5000`
- API：`POST /api/detect`

## 代码地图
### `app.py`
应用入口：
- 读取配置（`detector/config.py:load_config`）
- 创建 Flask app（`detector/webapp.py:create_app`）
- 提供 `app` 变量，方便 WSGI 部署（例如 `gunicorn app:app`）

### `detector/config.py`
集中配置定义与加载：
- `AppConfig` / `ModelConfig` / `CaptureConfig`
- 支持使用环境变量覆盖（便于部署/调参）

### `detector/yolo_detector.py`
模型推理层：
- 加载 YOLO 模型（吸烟必需，口罩可选）
- 把 ultralytics 的输出统一成 `DetectionResult`

### `detector/capture_policy.py`
证据保存策略层：
- 连续命中阈值（减少误报）
- 保存冷却（避免刷盘）
- 长事件重保存（同一持续违规周期性补证据）
- 文件轮转（超过上限删最旧）

### `detector/image_io.py`
输入/输出图像编解码：
- 从 `multipart/form-data(image)` 或 `image_data(DataURL)` 解码到 BGR
- 把标注后的 BGR 编码为 DataURL 返回给前端展示

### `detector/webapp.py`
Web 层（Flask 路由）：
- `/` 返回页面
- `/api/detect` 完成“解码 → 推理 → 标注 → 保存策略 → JSON 返回”

## 目录说明
- `templates/`：网页模板（当前只有 `index.html`）
- `static/`：静态资源（目前为空）
- `captures/`：违规截图输出（已在 `.gitignore` 忽略）
- `datasets/`：训练数据（体积大，建议按需管理/不要频繁提交）
- `runs/`：训练/推理输出（体积大，建议忽略或迁移到外部存储）

