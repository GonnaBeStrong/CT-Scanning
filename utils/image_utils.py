import os
from datetime import datetime
from pathlib import Path


def ensure_date_folder_exists(base_path: str) -> str:
    """确保当天日期文件夹存在，返回文件夹路径"""
    today = datetime.now().strftime("%Y-%m-%d")
    date_folder = Path(base_path) / today
    date_folder.mkdir(parents=True, exist_ok=True)
    return str(date_folder)


def generate_unique_filename(folder_path: str, original_name: str) -> str:
    """生成唯一文件名避免重复"""
    counter = 1
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix

    while True:
        new_name = f"{stem}_{counter}{suffix}" if counter > 1 else original_name
        if not (Path(folder_path) / new_name).exists():
            return new_name
        counter += 1