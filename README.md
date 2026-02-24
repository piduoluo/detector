# 厨房吸烟/口罩检测（YOLOv8 + Flask 网页端）

一个本地运行的网页应用：浏览器调用摄像头，将画面帧上传到 Flask 后端做 YOLO 推理，实时标注并在检测到违规（吸烟 / 未戴口罩）时自动截图保存。

## 功能
- 网页端实时检测（约每 0.8 秒一次，前端可控）
- 违规证据自动截图保存到 `captures/`（带连续命中阈值、冷却时间、数量上限与轮转删除）
- 吸烟模型：`smoking.pt`
- 口罩模型（可选）：`trained_mask_model.pt`（不存在时仅启用吸烟检测）

## 快速开始
### 1) 安装依赖
```bash
pip install -r requirements.txt
```

### 2) 启动服务
```bash
python app.py
```

### 3) 打开页面
浏览器访问：`http://127.0.0.1:5000`

## 使用说明
- 点击“开启摄像头”后开始实时检测
- 点击“停止检测”可暂停
- 违规截图保存在 `captures/`，文件名示例：`violation_20260220_143000_no-mask.jpg`

> 注：`captures/` 已在 `.gitignore` 中忽略，不会被提交到仓库。

## API
### `POST /api/detect`
支持两种输入：
- `multipart/form-data`：字段名 `image`
- `application/x-www-form-urlencoded`：字段名 `image_data`（DataURL，如 `data:image/jpeg;base64,...`）

返回 JSON 包含：
- `preview_image`：标注后画面的 DataURL（用于前端预览）
- `violation_detected` / `violation_key` / `violation_streak`
- `screenshot_saved` / `just_saved` / `screenshot_path`
- `mask_model_loaded`：是否加载到口罩模型

## 配置（后端）
可在 `app.py` 顶部按需调整：
- `SMOKING_CONF_THRESHOLD`、`MASK_CONF_THRESHOLD`
- `SAVE_STREAK_THRESHOLD`（连续命中次数）
- `SAVE_COOLDOWN_SECONDS`（保存冷却）
- `LONG_EVENT_RESAVE_SECONDS`（长事件周期性补存证据）
- `MAX_CAPTURE_FILES`（最大截图数）

## 本地摄像头脚本（可选）
`kitchen.py` 提供一个直接用 OpenCV 打开摄像头进行检测并用 macOS `say` 播报警的脚本：
```bash
python kitchen.py
```
按 `q` 退出窗口。

## 训练（可选）
- 数据集配置：`smoking.yaml`（默认路径 `datasets/smoking`）
- 训练脚本：`trainsmoking.py`

```bash
python trainsmoking.py
```

