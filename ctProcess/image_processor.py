# image_processor.py

import base64
from io import BytesIO
import os
import json
import time
from datetime import datetime
from PIL import Image, ImageDraw
import numpy as np
import torch
import logging
from utils.image_utils import ensure_date_folder_exists

logger = logging.getLogger(__name__)
def process_image_data(image_data , model):
    start = time.time()
    logger.info("------开始检测图片------")
    # 解码 Base64 字符串为图像数据
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))

    # 确保图像是 RGB 格式
    image = image.convert("RGB")

    # 调整图像大小为 640x640（或者其他符合 32 的倍数的尺寸）
    image = image.resize((640, 640))

    # 将图像转换为 numpy 数组
    img_array = np.array(image)

    # 归一化图像数据到 [0, 1]
    img_array = img_array / 255.0

    # 将 numpy 数组转换为 PyTorch 张量，并改变维度
    img_tensor = torch.from_numpy(img_array).float()
    img_tensor = img_tensor.permute(2, 0, 1)  # 转换为 CxHxW 格式
    img_tensor = img_tensor.unsqueeze(0)  # 添加 batch 维度

    # 使用 YOLO 模型进行检测
    with torch.no_grad():
        results = model.predict(img_tensor)

    detection_info = []

    # 遍历检测结果并绘制检测框
    # results 是一个包含预测结果的列表
    for result in results:
        # result.boxes 包含了框的信息
        for box in result.boxes:
            # 获取框的坐标 (x1, y1, x2, y2), 置信度，类别
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # 获取框的坐标
            confidence = box.conf[0].item()  # 获取置信度
            class_id = int(box.cls[0].item())  # 获取类别 ID
            addCTRecords(class_id);

            # 将检测框画到原图上
            draw = ImageDraw.Draw(image)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"Conf: {confidence:.2f}, Class: {class_id}", fill="red")

            detection_info.append({
                "confidence": confidence,
                "class_id": class_id
            })

    # 1. 确保当日文件夹存在
    date_folder = ensure_date_folder_exists("./download")

    # 保存处理后的图片到文件夹
    file_name = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    file_path = os.path.join(date_folder, file_name)

    # 保存图片
    image.save(file_path)
    logger.info("文件保存在%s",file_path)

    # 将处理后的图片保存到字节流中
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    end = time.time()
    logger.info("------图片检测完成,耗时%f秒------", start - end)

    # 将检测框信息转化为 JSON 字符串
    detection_info_json = json.dumps(detection_info)

    return detection_info_json, img_byte_arr.read()


def addCTRecords(class_id):
    # 读取文件
    with open('./records/ctRecords.txt', 'r') as file:
        content = file.read().strip()  # 读取内容并去除首尾空白
        numbers = [int(num) for num in content.split()]  # 分割并转换为数字

    numbers[0] += 1

    if class_id == 0:
        numbers[1] += 1
    elif class_id == 1:
        numbers[2] += 1
    elif class_id == 2:
        numbers[4] += 1
    elif class_id == 3:
        numbers[3] += 1

    # 将修改后的数字写回文件
    with open('./records/ctRecords.txt', 'w') as file:
        file.write(' '.join(map(str, numbers)))
