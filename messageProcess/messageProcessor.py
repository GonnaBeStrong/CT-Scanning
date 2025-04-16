import logging
import time

logger = logging.getLogger(__name__)
def processMessage(input_message, model_lange, tokenizer):
    start = time.time()
    logger.info("------开始回答问题------")
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
    response = response[0].split("### Response:")[1][12:-19]

    end = time.time();
    logger.info("------回答问题结束，耗时 %f s------", (end - start))

    return response
