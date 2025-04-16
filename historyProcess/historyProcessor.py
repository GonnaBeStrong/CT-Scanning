import os
from datetime import datetime
from http.client import HTTPException
import logging
logger = logging.getLogger(__name__)
def processHistory(date:str):
    logger.info("------开始获取历史检测信息-------")
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
    logger.info("------获取历史检测信息完毕------")
    return images
