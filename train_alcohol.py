from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data=r"Alcohol Bottle Detection.v1i.yolov8 (1)\data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    name="alcohol_detector"
)