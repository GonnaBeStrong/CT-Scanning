# 基于YOLOv8的医学影像检测系统（以脑部肿瘤为主）
## 项目简介
本项目为基于YOLOv8深度学习模型的医学影像检测系统，旨在辅助医学从业人员对医学影像异常目标的诊断判断，提高人们对医学影像的诊断效率。项目组成如下



**1. 检测模型**：使用YOLOv8深度学习模型，用于检测异常目标。

**2. 大语言模型**: **DeepSeek-R1-Distill-Qwen-1.5B** 作为底座模型，并基于医学知识数据集进行微调。  

**3. 后端**：使用Fast Api快速开发后端接口。

**4. 前端**：使用VUE3开发前端界面，使用Axios向后端发送请求。



## 项目架构
### 1. 整体架构
+ 后端：使用FastApi开发服务器接口，向前端提供各种功能



+ 前端：使用VUE3快速开发前端界面，向后端请求服务



+ 检测模型：基于 Ultralytics YOLOv8 预训练模型，可使用自定数据集训练。可对上传图像进行检测和判断，输出异常目标类别、置信度和边界框。



+ 大语言模型：**DeepSeek-R1-Distill-Qwen-1.5B** 作为底座模型，并基于医学知识数据集进行微调  

### 2. 技术栈
+ **后端框架：FastApi**
+ **前端框架：VUE3**
+ **目标检测模型：YOLOv8x-p6**
+ **大语言模型基座：DeepSeek-R1-Distill-Qwen-1.5B**
+ **大模型加速训练工具：unsloth**
+ **深度学习框架：PyTorch2.6**
+ **其它：** websocket进行数据传输



## 项目功能概述
1. 异常目标检测
    - 对三种的脑部肿瘤扫描进行检测判断。
    - 用户可自定义数据集进行训练。
    - 高精确度检测。



2. 智能问诊
    - 用户可提出自己个性化的问题，大模型将给出关于用户的患病状况的针对建议。



3. 结果可视化
    - 前端页面对模型预测结果进行展示，包括置信度、类别标签、目标边界框，用户可同时对比模型预测结果与上传原图。



4. 数据看板
    - 统计总检测数目，各类病症发现数目，用户反馈等


4. 历史记录
    - 用户可根据日期查看历史检测记录




## 项目安装部署
### 1. 环境要求
+ python >=3.8
+ pytorch >=1.8



### 2. 安装步骤
+ 模型环境准备安装 `ultralytics` 包，包括所有依赖项，终端输入以下代码



+ 部署后端项目

```plain
拉取项目至本地
```

```plain
cd 本项目根目录 //进入本项目的根目录
```

```plain
pip install --no-cache-dir -r requirements.txt // 下载本项目所需要的依赖
```


```plain
uvicorn main:app --host 127.0.0.1 --port 8000 --reload //启动后端服务
```



+ 部署前端项目（前端项目地址如下）

```plain
https://github.com/GonnaBeStrong/CT-scanning-foreend
```


## 后端项目结构解析

+	download文件夹，后端系统在检测完前端用户传来的图片之后，会将检测后的图片在download文件夹中以日期为分隔存储下来
+	commentProcess文件夹中放置用来处理用户评价的代码文件
+	ctProcess文件夹放置处理CT检测相关的代码文件
+	dataProcess文件夹放置处理数据看板有关的代码文件
+	historyProcess文件夹放置处理历史检测记录有关的代码文件
+	messageProcess文件夹放置处理AI问答系统有关的代码文件
+	LOG存放日志系统代码
+	logs文件夹放置系统生成的日志文件
+	records文件中记录图片检测结果数据，用户对AI问答系统的评价，为数据看板功能提供服务
+	ultralytics为程序提供必要的依赖
+	utils文件中存储一些实现功能所需要的工具类
+	main.py为后端服务的启动文件


## 后端接口文档

+ 部署前端项目（前端项目地址如下）

```plain
@app.websocket("/test-tumor")
前端通过该接口与后端系统建立websocket连接，将图片传送给后端，后端会将处理之后的图片通过websocket回送给前端
```

```plain
@app.post("/process_message")
前端通过该接口向后端系统发送用户提问的文本，该接口会以文本的形式将AI问答系统的回复发送给前端
```

```plain
@app.websocket("/ws/audio")
前端通过该接口向后端发送音频数据，后端将识别结果返回给前端
```

```plain
@app.get("/query-data")
前端通过该接口得到数据看板的统计数据，为两个数组
```

```plain
@app.get("/comment")
前端通过该接口向后端发送用户对AI问答系统的评价，后端进行相应的记录
```



## 关于模型训练
### 1. 数据集
1. 目标检测模型

来源：[https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset/dataset/1](https://universe.roboflow.com/ali-rostami/labeled-mri-brain-tumor-dataset/dataset/1)

数据集包括2500张MRI扫描图片，包括三种不同的肿瘤和无肿瘤四种情况的图片



2. 大语言模型

来源：[https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data-Chinese](https://huggingface.co/datasets/FreedomIntelligence/Medical-R1-Distill-Data-Chinese)

数据集包括17000多条医学问答，涵盖多领域。本项目截取2500条进行微调。



### 2. 模型参数
**目标检测模型**

```plain
model = YOLO("ultralytics/cfg/models/v8/yolov8x-p6.yaml").load("yolov8x.pt")

results = model.train(
    data='datasets/brain2/data.yaml',
    epochs=150,
    batch=28,
    imgsz=1024,
    freeze=10,
    lr0=0.002,
    lrf=0.01,
    hsv_h=0.005,  # 颜色增强
    hsv_s=0.5,  # 饱和度增强
    hsv_v=0.3,  # 适当亮度增强
    degrees=5.0,  # 旋转
    translate=0.05,  # 平移
    scale=0.3,  # 缩放
    shear=0.5,  # 降低错切
    flipud=0.3,  # 垂直翻转
    fliplr=0.5,  # 水平翻转
    mosaic=1.0,  # 保持 Mosaic
    mixup=0.1,  # MixUp
    copy_paste=0.05  # Copy-Paste
    )
    
    #说明
    #model=YOLO().load() 载入模型设置和预训练权重
    #model.train() 训练YOLO模型，其中训练参数可根据需要修改
    #data 指定数据集配置文件路径
    #epoch 指定模型训练代数，训练更多代可能会提高模型性能，需要一定数量的代数才可使效果呈现收敛
    #batch 指定每批训练的图片数，大批量可加速训练，但要求更多GPU显存
    #imgsz 指定图像的大小，更高的分辨率可能会提高模型性能，但会增加计算开销
    #freeze 指定冻结预训练权重的层数，冻结一定层数可避免模型收敛过早
    #lr0 指定初始学习率，设定小的值可避免模型收敛过早
    #lrf 指定模型最终的学习率
    #hsv_h, degrees等参数为数据增强参数，用于扩充训练数据
```

参数详情参考官方文档  [https://docs.ultralytics.com/zh/usage/cfg/](https://docs.ultralytics.com/zh/usage/cfg/)





**大语言模型**

```plain
# 使用FastLanguageModel工具将基础模型转换为PEFT(Parameter-Efficient Fine-Tuning)模型
# 主要采用LoRA(Low-Rank Adaptation)技术进行高效微调
model = FastLanguageModel.get_peft_model(
  model, # 基础语言模型（如Llama、Mistral等）
# LoRA核心参数
  r=16, # LoRA矩阵的秩（rank），决定可训练参数数量
      # 典型值：8-64，值越大训练参数越多，16是平衡效果与效率的常用选择

# 指定应用LoRA的模块（针对LLaMA架构的典型配置）

  target_modules=[
    "q_proj", # 查询(Query)投影层
    "k_proj", # 键(Key)投影层
    "v_proj", # 值(Value)投影层
    "o_proj", # 输出(Output)投影层
    "gate_proj", # 门控(Gate)投影层（FFN第一部分）
    "up_proj", # 上投影层（FFN第二部分）
    "down_proj", # 下投影层（FFN第三部分）
  ],

# 注意：不同模型架构需要调整target_modules

  lora_alpha=16, # LoRA缩放因子（控制新参数对原始参数的权重）
          # 通常设置为与r相同或2倍的值
  lora_dropout=0, # LoRA层的dropout率（0表示不使用）
          # 典型值：0-0.2，设为0可获得更稳定训练
  bias="none", # 偏置项处理方式：
          # "none"：不训练任何偏置参数（最常用）
          # "all"：训练所有偏置
          # "lora_only"：仅训练LoRA部分的偏置
# 梯度检查点配置（内存优化技术）
  use_gradient_checkpointing="unsloth", # 使用Unsloth优化的检查点技术
                      # 特别适合长上下文训练（7K+ tokens）
                      # 可用True启用普通检查点
  random_state=3407, # 随机种子（确保LoRA初始化可复现）
            # 3407是常用"魔法种子"
  use_rslora=False, # 是否使用rsLoRA（Rescaled LoRA）变体
            # False表示标准LoRA
  loftq_config=None, # LoFTQ量化配置（None表示不应用）
            # 可用于量化感知训练
)
```



```plain
trainer = SFTTrainer(
  # 基础模型配置
  model=model, # 要微调的模型（通常是PeftModel或基础LLM）
  tokenizer=tokenizer, # 与模型匹配的分词器
  # 数据集配置
  train_dataset=train_dataset, # 训练数据集（需包含格式化后的文本）
  eval_dataset=eval_dataset,
  dataset_text_field="text", # 指定数据集中包含文本的字段名
  max_seq_length=2048, # 最大序列长度（如2048或4096）
  dataset_num_proc=2, # 数据集预处理进程数（加速数据加载）

  # 训练参数配置
  args=TrainingArguments(
    per_device_train_batch_size=8,          # 每个GPU的 batch 大小
    gradient_accumulation_steps=2,          # 累积4步，相当于batch=8

    num_train_epochs=2,                     # 训练轮数（可改为3~5）
    warmup_steps=100,                       # 预热步数，可占总训练步数的5~10%
    learning_rate=2e-4,                     # 学习率（适合LoRA）
    weight_decay=0.01,                      # 权重衰减，防过拟合

    # 优化器配置
    #learning_rate=2e-4, # 学习率（LoRA常用2e-4到5e-4）
    optim="adamw_8bit", # 8bit AdamW优化器（节省显存）
    #weight_decay=0.01, # 权重衰减（防止过拟合）

    # 精度配置
    fp16=not is_bfloat16_supported(), # 自动启用fp16（如果bfloat16不可用）
    bf16=is_bfloat16_supported(), # 优先使用bfloat16（A100/TPU等支持）

    save_strategy="epoch",                  # 每轮保存模型
    evaluation_strategy="epoch",           # 每轮进行验证（需提供 eval_dataset）
    save_total_limit=2,                     # 最多保存2个 checkpoint
    load_best_model_at_end=True,            # 自动加载验证集表现最好的模型

    # 日志与输出
    lr_scheduler_type="linear", # 线性学习率调度
    seed=3407, # 随机种子（确保可复现性）
    output_dir="outputs", # 模型和日志输出目录
    report_to="tensorboard",                # 记录日志到TensorBoard

  ),
)
```

详情参考官方文档

[https://docs.unsloth.ai/](https://docs.unsloth.ai/)

[https://huggingface.co/docs/transformers/v4.17.0/en/index](https://huggingface.co/docs/transformers/v4.17.0/en/index)


