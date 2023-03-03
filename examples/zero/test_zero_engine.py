#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from functools import partial

import colossalai
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from colossalai.core import global_context as gpc
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.sharded_model.utils import col_model_deepcopy
from colossalai.zero.sharded_optim._utils import has_inf_or_nan
from tests.components_to_test.registry import non_distributed_component_funcs
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models import resnet18
from common import (MP_PARALLEL_CONFIG, ZERO_PARALLEL_CONFIG, check_params, check_sharded_model_params)
"""
参考 https://github.com/hpcaitech/ColossalAI/blob/v0.2.5/tests/test_zero/test_zero_engine.py
"""

def run_dist(rank, world_size, port, parallel_config):
    colossalai.launch(config=parallel_config,
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl')


    # ----------------------------------------------------------------
    train_dataloader = get_cifar10_dataloader(train=True)
    criterion=torch.nn.CrossEntropyLoss()
    # ----------------------------------------------------------------

    with ZeroInitContext(target_device=torch.cuda.current_device(),
                         shard_strategy=gpc.config.zero.model_config.shard_strategy,
                         shard_param=True):
        colo_model = resnet18(num_classes=10)

    colo_optimizer = torch.optim.Adam(colo_model.parameters(), lr=1e-3)
    engine, train_dataloader, _, _ = colossalai.initialize(colo_model,
                                                           optimizer=colo_optimizer,
                                                           criterion=criterion,
                                                           train_dataloader=train_dataloader)

    # torch_model = resnet18(num_classes=10).half()
    # col_model_deepcopy(engine.model, torch_model) # 将engine.model的fp16权重赋值给torch_fp16模型
    # torch_model = torch_model.cuda().float() # torch模型权重转为正常的fp32

    engine.train()
    # torch_optimizer = optimizer_class(torch_model.parameters(), lr=1e-3)
    # 将torch转为分布式数据并行
    # if dist.get_world_size() > 1:
    #     torch_model = DDP(torch_model, device_ids=[torch.cuda.current_device()])


    for data, label in train_dataloader:


        data, label = data.cuda(), label.cuda()

        engine.zero_grad()
        # torch_optimizer.zero_grad()


        output = engine(data)
        loss = engine.criterion(output, label)

        # torch_output = torch_model(data)
        # torch_loss = engine.criterion(torch_output, label)


        engine.backward(loss)
        engine.step()

        # torch_loss.backward()

        # torch优化器更新  判断fp16模型梯度 不为 空或无穷大
        # for param in torch_model.parameters():
        #     if param.grad is not None:
        #         assert not has_inf_or_nan(param.grad)
        # torch_optimizer.step()

    # 比较两个模型的权重 是否一致
    # if parallel_config == ZERO_PARALLEL_CONFIG:
    #     check_sharded_model_params(torch_model, colo_model, loose=True)




@pytest.mark.dist
@pytest.mark.parametrize("world_size", [1, 2])
@rerun_if_address_is_in_use()
def test_zero_engine(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), parallel_config=ZERO_PARALLEL_CONFIG)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_zero_engine(world_size=4)
