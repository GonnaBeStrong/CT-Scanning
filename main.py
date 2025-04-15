import base64
import os
import uvicorn
from datetime import datetime
import json
from http.client import HTTPException
from pydub import AudioSegment
from pydub.utils import which
from starlette.staticfiles import StaticFiles

from utils.image_utils import ensure_date_folder_exists

from pydantic import BaseModel
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

import numpy as np
import torch
import time
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from io import BytesIO
from PIL import Image, ImageDraw
import io


# 拉取大模型（请在这里填入自己的模型拉取地址）
start_time = time.time()
login(token="xxx")
tokenizer = AutoTokenizer.from_pretrained("bubblepot/R1_test2")
model_lange = AutoModelForCausalLM.from_pretrained("bubblepot/R1_test2")
middle_time = time.time()
print("模型已经下载完成,耗时：", middle_time - start_time, "秒")

# 加载肿瘤识别模型（请在这里填入自己的肿瘤识别模型文件路径）
model = YOLO('p6_150.pt')
print("模型已经下载完成,耗时：", middle_time - start_time, "秒")

# 启动应用
app = FastAPI()
# 挂载图片目录（方便直接访问）
app.mount("/download", StaticFiles(directory="./"), name="static")
# 配置跨域请求，支持前后端分离
origins = [
    "http://localhost:8080",  # 允许的前端地址（例如 Vue 前端）
    "http://127.0.0.1:8080",  # 允许的前端地址
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 允许跨域请求的源
    allow_credentials=True,  # 允许携带凭证（如 Cookie）
    allow_methods=["*"],  # 允许所有 HTTP 方法
    allow_headers=["*"],  # 允许所有 HTTP 请求头
)


@app.websocket("/test-tumor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 接收客户端发送的 Base64 编码的图片数据
            try:
                image_data = await websocket.receive_text()
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
                date_folder = ensure_date_folder_exists("./download/")

                # 保存处理后的图片到文件夹
                file_name = f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                file_path = os.path.join(date_folder, file_name)

                # 保存图片
                image.save(file_path)
                print(f"Image saved at {file_path}")

                # 将处理后的图片保存到字节流中
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)

                # 将检测框信息转化为 JSON 字符串
                detection_info_json = json.dumps(detection_info)

                # 通过 WebSocket 返回处理后的图片
                await websocket.send_text(detection_info_json)
                await websocket.send_bytes(img_byte_arr.read())
            except Exception as e:
                # 图像解析失败
                print("格式错误")
                await websocket.send_text(json.dumps({
                    "status": 401,
                }))
                return

    except WebSocketDisconnect:
        print("Client disconnected")



# 定义请求体的模型，接收一段话
class Message(BaseModel):
    content: str

@app.post("/process_message")
async def process_message(message: Message):
    # # 获取前端传来的话
    input_message = message.content
    print(input_message)
    # 假设这里进行了一些处理，比如把所有字母转为大写
    prompt_style = """以下是一项任务的描述，附带提供背景信息的输入内容。
    请撰写能恰当完成该请求的响应。在回答前，请仔细思考问题并建立分步推理链条，以确保回答的逻辑性和准确性。
    ### Instruction:
    您是一位资深的医学专家、博士，在医学诊断，疾病治疗，愈后康复方面拥有深厚的专业知识。 请回答以下医学问题。
    ### Question:
    {}
    ### Response:
    <think>{}"""

    question = input_message

    inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt")
    outputs = model_lange.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1200,
        use_cache=True,
    )
    response = tokenizer.batch_decode(outputs)
    print(response[0].split("### Response:")[1])
    response = response[0].split("### Response:")[1][12:-19];


    # 返回处理后的消息
    return JSONResponse(content={"content": response})


# @app.websocket("/ws/audio")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         # 接收来自前端的音频数据
#         audio_data = await websocket.receive_bytes()
#
#         print("ffmpeg path:", which("ffmpeg"))
#         print("ffprobe path:", which("ffprobe"))
#         AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
#         audio = AudioSegment.from_wav(io.BytesIO(audio_data))
#         audio = audio.set_frame_rate(16000)
#         audio.export('myPcm.pcm', format="raw")
#
#         # 在这里，你可以处理音频数据（例如：保存、分析、转发等）
#
#         # 回复客户端（可以选择是否要发送回应）
#         await websocket.send_text("Audio data received")

@app.get("/query-data")
async def process_message():
    # 读取文件
    with open('./records/ctRecords.txt', 'r') as file:
        content = file.read().strip()  # 读取内容并去除首尾空白
        numbersCT = [int(num) for num in content.split()]  # 分割并转换为数字

    with open('./records/ansRecords.txt', 'r') as file:
        content = file.read().strip()  # 读取内容并去除首尾空白
        numbersANS = [int(num) for num in content.split()]  # 分割并转换为数字


    # 返回处理后的消息
    return JSONResponse(content={"numbersCT": numbersCT,"numbersANS": numbersANS})


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


@app.get("/comment")
async def comment(commentType: int):  # 必填整数参数
    # 读取文件
    with open('./records/ansRecords.txt', 'r') as file:
        content = file.read().strip()  # 读取内容并去除首尾空白
        numbers = [int(num) for num in content.split()]  # 分割并转换为数字

    numbers[0] += 1

    if commentType == 1:
        numbers[1] += 1
    elif commentType == 2:
        numbers[2] += 1
    elif commentType == 3:
        numbers[3] += 1
    elif commentType == 4:
        numbers[4] += 1

    # 将修改后的数字写回文件
    with open('./records/ansRecords.txt', 'w') as file:
        file.write(' '.join(map(str, numbers)))


# 获取指定日期的所有图片
@app.get("/images")
async def get_images_by_date(date: str):
    try:
        # 验证日期格式
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    image_dir = "./download/" + date
    if not os.path.exists(image_dir):
        return {"images": []}

    # 获取目录下所有图片文件
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 返回图片URL列表
    return {
        "images": [f"http://localhost:8000/download/download/{date}/{img}" for img in images],
        "date": date
    }








