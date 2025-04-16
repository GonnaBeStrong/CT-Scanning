import base64
import os
import uvicorn
import json
import logging
import numpy as np
import torch
import time
import io
from io import BytesIO
from PIL import Image, ImageDraw
from http.client import HTTPException
from pydub import AudioSegment
from pydub.utils import which
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.staticfiles import StaticFiles
from commentProcess.commentProcessor import processComment
from ctProcess.image_processor import process_image_data
from dataProcess.dataProcessor import processData
from historyProcess.historyProcessor import processHistory
from messageProcess.messageProcessor import processMessage
from utils.image_utils import ensure_date_folder_exists
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from LOG.log_config import setup_logger


# 初始化日志系统
setup_logger()
logger = logging.getLogger(__name__)
logger.info("----------项目启动------------")

# 拉取大模型（请在这里填入自己的模型拉取地址）
start_time = time.time()
login(token="xxx")
tokenizer = AutoTokenizer.from_pretrained("bubblepot/R1_test2")
model_lange = AutoModelForCausalLM.from_pretrained("bubblepot/R1_test2")
middle_time = time.time()
logger.info("AI问答模型已经下载完成,耗时：%d 秒", middle_time - start_time)

# 加载肿瘤识别模型（请在这里填入自己的肿瘤识别模型文件路径）
model = YOLO('p6_150.pt')
last_time = time.time()
logger.info("CT检测模型已经加载完成,耗时：%d 秒", last_time - middle_time)

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


# 检测CT图片接口
@app.websocket("/test-tumor")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            # 接收客户端发送的 Base64 编码的图片数据
            try:
                image_data = await websocket.receive_text()
                detection_info_json, image_bytes = process_image_data(image_data, model)
                await websocket.send_text(detection_info_json)
                await websocket.send_bytes(image_bytes)
            except Exception as e:
                # 图像解析失败
                logger.error("图片格式错误，返回错误码")
                await websocket.send_text(json.dumps({
                    "status": 401,
                }))
                return

    except WebSocketDisconnect:
        print("Client disconnected")


# 定义请求体的模型，接收一段话
class Message(BaseModel):
    content: str

# AI问答系统接口
@app.post("/process_message")
async def process_message(message: Message):
    # 获取前端传来的话
    input_message = message.content
    # 回答问题
    response = processMessage(input_message, model_lange, tokenizer)
    # 返回处理后的消息
    return JSONResponse(content={"content": response})


# 数据看板功能接口
@app.get("/query-data")
async def process_message():
    numbersCT, numbersANS = processData()
    # 返回处理后的消息
    return JSONResponse(content={"numbersCT": numbersCT, "numbersANS": numbersANS})

# 用户评论功能接口
@app.get("/comment")
async def comment(commentType: int):  # 必填整数参数
    processComment(commentType)


# 历史记录功能接口
@app.get("/images")
async def get_images_by_date(date: str):
    images = processHistory(date)
    # 返回图片URL列表
    return {
        "images": [f"http://localhost:8000/download/download/{date}/{img}" for img in images],
        "date": date
    }
