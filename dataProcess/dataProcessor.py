import logging
logger = logging.getLogger(__name__)
def processData():
    logger.info("------开始处理数据看板------")
    # 读取文件
    with open('./records/ctRecords.txt', 'r') as file:
        content = file.read().strip()  # 读取内容并去除首尾空白
        numbersCT = [int(num) for num in content.split()]  # 分割并转换为数字

    with open('./records/ansRecords.txt', 'r') as file:
        content = file.read().strip()  # 读取内容并去除首尾空白
        numbersANS = [int(num) for num in content.split()]  # 分割并转换为数字
    logger.info("------处理数据看板结束------")
    return numbersCT, numbersANS
