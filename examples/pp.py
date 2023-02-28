import os
from typing import Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
import colossalai
import colossalai.nn as col_nn

from colossalai.core import global_context as gpc
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer, get_dataloader
from colossalai.context import ParallelMode
from colossalai.pipeline.pipelinable import PipelinableContext

from titans.dataloader.cifar10 import build_cifar
from torchvision.models import resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

"""
参考https://colossalai.org/zh-Hans/docs/features/pipeline_parallel

第二个例子  https://github.com/hpcaitech/ColossalAI/blob/v0.2.5/examples/tutorial/hybrid_parallel/train.py
"""

# ---------------------------参数配置-------------------------------------
# Define some config
BATCH_SIZE = 64  # 总batch
NUM_EPOCHS = 2  # 总训练轮数
NUM_CHUNKS = 1
CONFIG = dict(
    NUM_MICRO_BATCHES=4, parallel=dict(pipeline=2)
)  # 流水并行：2个流水段  batch被切分为4个micro batches
# NUM_MICRO_BATCHES + NUM_CHUNKS = 使用交错schedule

# Train
colossalai.launch_from_torch(config=CONFIG)
logger = get_dist_logger()
# ---------------------------构建模型-------------------------------------
pipelinable = PipelinableContext()  # 流水线上下文管理器，将模型切分成流水阶段

with pipelinable:
    model = resnet50()  # torch自带的普通模型

# 给定切分顺序 model.module得到模型序列
exec_seq = [
    "conv1",
    "bn1",
    "relu",
    "maxpool",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "avgpool",
    (lambda x: torch.flatten(x, 1), "behind"),  # 拉伸操作 需手动添加
    "fc",
]
pipelinable.to_layer_list(exec_seq)

# 将模型切分成流水线阶段
model = pipelinable.partition(
    NUM_CHUNKS, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE)
)
print("gpc.pipeline_parallel_size ", gpc.pipeline_parallel_size)  # 2
print("ParallelMode.PIPELINE ", ParallelMode.PIPELINE)  # ParallelMode.PIPELINE
print("--------------------------------")
print(model)  # 查看划分后的模型结构   PipelinableModel对象

# ------------------------------训练----------------------------------
# build criterion
criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# build dataloader
root = "./data"  # 指定数据集保存路径
train_dataloader, test_dataloader = build_cifar(
    BATCH_SIZE, root, padding=4, crop=32, resize=32
)
# colosalai内置的阶梯式学习率调度器（与并行策略无关）
lr_scheduler = col_nn.lr_scheduler.LinearWarmupLR(optimizer, NUM_EPOCHS, warmup_steps=1)
engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(
    model, optimizer, criterion, train_dataloader, test_dataloader, lr_scheduler
)
timer = MultiTimer()

trainer = Trainer(engine=engine, timer=timer, logger=logger)

hook_list = [
    hooks.LossHook(),
    hooks.AccuracyHook(col_nn.metric.Accuracy()),
    hooks.LogMetricByEpochHook(logger),
    hooks.LRSchedulerHook(lr_scheduler, by_epoch=True),
]

trainer.fit(
    train_dataloader=train_dataloader,
    epochs=NUM_EPOCHS,
    test_dataloader=test_dataloader,
    test_interval=1,
    hooks=hook_list,
    display_progress=True,
)


# 启动命令
# colossalai run --nproc_per_node 2 pp.py
