from ultralytics import YOLO

model = YOLO("D:\interesting_projects/ultralytics/yolo11n.pt", task="detect")
# model = YOLO(r"D:\yolo11\runs\detect\train\weights/best.pt", task="detect")
# path = model.export(format="onnx", simplify=True, device=0, opset=16, dynamic=False, imgsz=640)
path = model.export(format="onnx", simplify=True, device=0, opset=11, dynamic=False, imgsz=640)