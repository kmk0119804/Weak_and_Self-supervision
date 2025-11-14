from ultralytics import YOLO
import os
# model_path = '/home/yeji/Desktop/research/best_204.pt'
model_path = '/home/yeji/runs/segment/train100/weights/best.pt' 
data_yaml  = '/home/yeji/Desktop/research/detection.yaml'
# /home/yeji/runs/segment/train74/weights
# '/home/yeji/Downloads/best.pt'
model = YOLO(model_path)
results = model.val(
    data=data_yaml,       # ✅ 여기! source 말고 data
    imgsz=1280,
    conf=0.25,
    # iou=0.5,              # 원하면 조정
    save_json=True,
    project='/home/yeji/Desktop/research',
    # name='val_best204',
    exist_ok=True
)




# from ultralytics import YOLO
# import os
# # model_path = '/home/yeji/Desktop/research/best_204.pt'
# model_path = '/home/yeji/runs/segment/train90/weights/best.pt'
# images_dir = '/home/yeji/Desktop/research/DATA_optimization/images/val'  # 실제 이미지 폴더

# model = YOLO(model_path)
# results = model.predict(
#     source=images_dir,     # ✅ predict는 source
#     imgsz=1280,
#     conf=0.25,
#     save=True,             # 결과 이미지 저장
#     save_txt=True,         # 라벨 txt 저장
#     save_conf=True,        # confidence 저장
#     save_json=True,                # COCO json 저장
#     project='/home/yeji/Desktop/research/pred_best',
#     name='val12',
#     exist_ok=True
# )
