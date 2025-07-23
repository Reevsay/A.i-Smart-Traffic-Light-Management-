from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='cpu',
    workers=4,
    augment=True,
    project='traffic_model',
    name='experiment1',
    val=True,
    optimizer='SGD',
    conf=0.3,
    iou=0.5
)

print("Training complete. Results saved to:", results.save_dir)
