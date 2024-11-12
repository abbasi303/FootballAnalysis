from ultralytics import YOLO
model= YOLO('models/best.pt')

results = model.predict('input_video/08fd33_4.mp4', save=True, project='runs/detect', name='output_video')
print(results[0])
print('=============================')
for box in results[0].boxes:
    print(box)