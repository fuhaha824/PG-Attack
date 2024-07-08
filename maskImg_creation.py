from PIL import Image, ImageDraw
from ultralyticsplus import YOLO, render_result
import os
# 加载模型
model = YOLO('ultralyticsplus/yolov8s')

# 设置模型参数
model.overrides['conf'] = 0.25  # NMS 置信度阈值
model.overrides['iou'] = 0.45  # NMS IoU 阈值
model.overrides['agnostic_nms'] = False  # 非类别不可知的 NMS
model.overrides['max_det'] = 1000  # 每张图像的最大检测数
output_folder = 'images_mask'
os.makedirs(output_folder, exist_ok=True)

for i in range(100):
    print(i)
    image_path = f'images-2/{str(i).zfill(6)}.jpg'
    results = model.predict(image_path)
    boxes = results[0].boxes.xyxy
    labels = [1] * len(boxes)
    image = Image.open(image_path)
    filled_image = Image.new("RGB", image.size, color="white")
    draw = ImageDraw.Draw(filled_image)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([(x1, y1), (x2, y2)], fill=(0,0,0))
        draw.text((x2, y1), str(int(label)), fill=(0,0,0))
    output_path = os.path.join(output_folder, f'{str(i).zfill(6)}.jpg')
    filled_image.save(output_path)
