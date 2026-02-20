# 厨房吸烟检测（网页端）

## 1. 安装依赖
```bash
pip install -r requirements.txt
```

## 2. 启动服务
```bash
python app.py
```

## 3. 打开页面
浏览器访问：
`http://127.0.0.1:5000`

## 4. 使用方式
- 点击“开启摄像头”后自动实时检测（约每 0.8 秒一次）
- 点击“停止检测”可暂停实时检测
- 首次检测到吸烟会自动截图并保存到 `captures/smoking_detected_once.jpg`
- 只会保存一张，后续检测到吸烟不会重复保存
