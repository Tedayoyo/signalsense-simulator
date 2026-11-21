# SignalSense edge prototype

The working core of SignalSense running in the browser: a YOLOv8 model watches junction footage, counts the vehicles waiting in each approach's zone, scores them in PCU (IRC 106-1990 weights) and plans each green from that score. Ambulance, fire-truck and school-bus alerts preempt or nudge the plan. Decisions are published as MQTT-style messages to a virtual controller (shadow mode).

Open `index.html` through a web server (GitHub Pages works). It will not run from a double-clicked file, because browsers block the model worker on `file://`. Locally: `python -m http.server` in this folder, then open http://localhost:8000.

## Files
- `index.html`: console, scoring and signal controller
- `detector.js`: Web Worker that runs the model (ONNX Runtime Web, WebAssembly)
- `model/yolov8n.onnx`: YOLOv8n (COCO) exported with Ultralytics, 640 px
- `ort/`: ONNX Runtime Web 1.22.0 runtime files
- `junction.mp4`: 15 s junction camera clip

## Using your own model
Train YOLOv8 on your Indian classes, then `yolo export model=best.pt format=onnx imgsz=640 opset=17` and load the `.onnx` with "Load my model". Class names are read from the file. Names containing ambulance / fire truck / police trigger preemption automatically; auto / rickshaw score 1.2 PCU; school bus gets tier-2 priority.

## Credits
- Footage: EFS TrafficMonitoring, https://github.com/EFS-OpenSource/TrafficMonitoring (Apache-2.0)
- Model: Ultralytics YOLOv8n, https://github.com/ultralytics/ultralytics (AGPL-3.0)
- Runtime: ONNX Runtime Web, https://github.com/microsoft/onnxruntime (MIT)
