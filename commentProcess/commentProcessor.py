import logging
logger = logging.getLogger(__name__)
def processComment(commentType):
    logger.info("------开始处理用户评价------")
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

    logger.info("------处理用户评价完成------")