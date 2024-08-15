from ultralytics import YOLO

model = YOLO(r"D:\Deep Learning Projects\YOLOv8 Project\Object Counting\Pedestrian_counting\detect\train\weights\best.pt")
source = r"vid1.mp4"

# Run inference on the source
model.predict(source=source, save=True, imgsz=320, conf=0.5)

