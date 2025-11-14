from ultralytics import YOLO
import os

# Create a new YOLO model from scratch
# # model = YOLO('/home/yeji/Desktop/research/best_204.pt')

# # Load a pretrained YOLO model (recommended for training)
# model = YOLO(model = '/home/yeji/Desktop/research/best_204.pt', task = 'segment')

# # Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='/home/yeji/Desktop/research/detection.yaml', cfg='/home/yeji/Desktop/research/ultralytics/ultralytics/yolo/cfg/default.yaml')

# # Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
# results = model('/home/yeji/Desktop/yolov5_data/test_jpg')

# Export the model to ONNX format
# success = model.export(format='onnx')

# current_dir = '/home/yeji/Desktop/research'

# file_path = os.path.join(current_dir, 'runs')
# print(file_path)





from ultralytics import YOLO

model = YOLO("/home/yeji/Desktop/research/best_204.pt")  # 예시
results = model.train(
    data="/home/yeji/Desktop/research/detection.yaml",
    epochs=300,
    imgsz=1280,        # ← 이미지 크기
    batch=6,         # ← 배치 크기
    device=0          # ← GPU 번호
)



# from ultralytics import YOLO

# # yolov8m-seg 모델 불러오기 (사전학습 weight)
# model = YOLO("yolov8m-seg.pt")

# # 학습 수행
# results = model.train(
#     data="/home/yeji/Desktop/research/detection.yaml",
#     epochs=300,
#     imgsz=1280,
#     batch=6,
#     device=0
# )





# import math
# from ultralytics import YOLO

# def set_ema_decay(trainer):
#     if getattr(trainer, "ema", None):
#         # updates 인자를 받는 callable로 만들어야 함
#         trainer.ema.decay = lambda updates: 0.9995
#         print("[Hook] EMA decay set to constant 0.9995")

# model = YOLO(model='/home/yeji/Desktop/research/best_204.pt', task='segment')
# model.add_callback("on_pretrain_routine_end", set_ema_decay)

# results = model.train(
#     data='/home/yeji/Desktop/research/detection.yaml',
#     # cfg ='/home/dlab4/Desktop/sam_mk/ultralytics/ultralytics/cfg/default.yaml',
#     epochs=300,
#     imgsz=1280,        # ← 이미지 크기
#     batch=6,         # ← 배치 크기
#     device=0          # ← GPU 번호
# )