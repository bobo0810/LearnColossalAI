#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import colossalai
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.sharded_model.utils import col_model_deepcopy
from colossalai.zero.sharded_optim._utils import has_inf_or_nan
from torch.nn.parallel import DistributedDataParallel as DDP
from resnet import get_resnet_training_components
from common import (
    MP_PARALLEL_CONFIG,
    ZERO_PARALLEL_CONFIG,
    check_params,
    check_sharded_model_params,
)

"""
参考 https://github.com/hpcaitech/ColossalAI/blob/v0.2.5/tests/test_zero/test_zero_engine.py
"""


def run_dist():

    colossalai.launch_from_torch(config=ZERO_PARALLEL_CONFIG)
    (
        model_builder,
        train_dataloader,
        _,
        optimizer_class,
        criterion,
    ) = get_resnet_training_components()
    with ZeroInitContext(
        target_device=torch.cuda.current_device(),
        shard_strategy=gpc.config.zero.model_config.shard_strategy,
        shard_param=True,
    ):
        colo_model = model_builder(checkpoint=True)

    colo_optimizer = optimizer_class(colo_model.parameters(), lr=1e-3)
    engine, train_dataloader, _, _ = colossalai.initialize(
        colo_model,
        optimizer=colo_optimizer,
        criterion=criterion,
        train_dataloader=train_dataloader,
    )
    # torch_model = model_builder(checkpoint=True).half()
    # col_model_deepcopy(engine.model, torch_model)  # 将engine.model的fp16权重赋值给torch_fp16模型
    # torch_model = torch_model.cuda().float()  # torch模型权重转为正常的fp32

    engine.train()
    # torch_optimizer = optimizer_class(torch_model.parameters(), lr=1e-3)
    # 将torch转为分布式数据并行
    # if dist.get_world_size() > 1:
    #     torch_model = DDP(torch_model, device_ids=[torch.cuda.current_device()])
    epoch = 3
    for i_epoch in range(epoch):
        print("epoch:{}/{}...".format(i_epoch, epoch))
        for data, label in train_dataloader:

            data, label = data.cuda(), label.cuda()

            engine.zero_grad()
            # torch_optimizer.zero_grad()

            if criterion:
                output = engine(data)
                loss = engine.criterion(output, label)

                # torch_output = torch_model(data)
                # torch_loss = engine.criterion(torch_output, label)
            else:
                loss = engine(data, label)
                # torch_loss = torch_model(data, label)

            engine.backward(loss)
            engine.step()

            # torch_loss.backward()

            # torch优化器更新  判断梯度 不为 空或无穷大
            # for param in torch_model.parameters():
            #     if param.grad is not None:
            #         assert not has_inf_or_nan(param.grad)

            # torch_optimizer.step()

    # 比较两个模型的权重 是否一致
    check_sharded_model_params(torch_model, colo_model, loose=True)


run_dist()
