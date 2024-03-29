import os

import torch
from titans.model.vit.vit import _create_vit_model # titans为colossalai内置模型库，已支持分布式
from tqdm import tqdm

import colossalai
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.nn import CrossEntropyLoss
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.utils import is_using_pp
from timm.utils import ModelEmaV2  # timm模型时需导入该包

class DummyDataloader():
    """合成假数据"""
    def __init__(self, length, batch_size):
        self.length = length
        self.batch_size = batch_size

    def generate(self):
        data = torch.rand(self.batch_size, 3, 224, 224)
        label = torch.randint(low=0, high=10, size=(self.batch_size,))
        return data, label

    def __iter__(self):
        self.step = 0
        return self

    def __next__(self):
        if self.step < self.length:
            self.step += 1
            return self.generate()
        else:
            raise StopIteration

    def __len__(self):
        return self.length


def main():
    # launch from torch
    parser = colossalai.get_default_parser()
    args = parser.parse_args()
    colossalai.launch_from_torch(config=args.config) # 加载config.py配置文件

    # get logger
    logger = get_dist_logger()
    logger.info("initialized distributed environment", ranks=[0])

    # 创建log文件
    if hasattr(gpc.config, 'LOG_PATH'):
        if gpc.get_global_rank() == 0:
            log_path = gpc.config.LOG_PATH
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            logger.log_to_file(log_path)

    # 检查配置文件是否满足流水线并行的要求
    use_pipeline = is_using_pp()

    # create model 定义模型的结构
    model_kwargs = dict(img_size=gpc.config.IMG_SIZE,
                        patch_size=gpc.config.PATCH_SIZE,
                        hidden_size=gpc.config.HIDDEN_SIZE,
                        depth=gpc.config.DEPTH,
                        num_heads=gpc.config.NUM_HEADS,
                        mlp_ratio=gpc.config.MLP_RATIO,
                        num_classes=10,
                        init_method='jax',
                        checkpoint=gpc.config.CHECKPOINT)
    # 已统一支持Tensor并行、数据并行



    # 开启流水并行
    if use_pipeline:
        # 流水线上下文管理器，将模型切分成流水阶段
        pipelinable = PipelinableContext()
        with pipelinable:
            model = _create_vit_model(**model_kwargs)
            # import timm
            # model = timm.create_model("convnext_xlarge_384_in22ft1k", pretrained=True, num_classes=10)
        pipelinable.to_layer_list() # colossalai默认将"模型初始化顺序"作为"切分顺序"

        # 通过分区策略构建分区模型 支持"balanced(默认)"、"uniform"
        # pipelinable.policy = "uniform"


        # 将模型切分成流水线阶段   num_chunks=1指交错式流水并行
        model = pipelinable.partition(1, gpc.pipeline_parallel_size, gpc.get_local_rank(ParallelMode.PIPELINE))
    else:
        model = _create_vit_model(**model_kwargs)
        # import timm
        # model = timm.create_model("convnext_xlarge_384_in22ft1k", pretrained=True, num_classes=10)

    # count number of parameters
    # 统计不同流水线阶段上的模型参数个数
    total_numel = 0
    for p in model.parameters():
        total_numel += p.numel()
    if not gpc.is_initialized(ParallelMode.PIPELINE):
        pipeline_stage = 0
    else:
        pipeline_stage = gpc.get_local_rank(ParallelMode.PIPELINE)
    logger.info(f"number of parameters: {total_numel} on pipeline stage {pipeline_stage}")

    # use synthetic dataset 使用伪造数据集
    # we train for 10 steps and eval for 5 steps per epoch
    # 训练集10次迭代   验证集5次迭代
    train_dataloader = DummyDataloader(length=10, batch_size=gpc.config.BATCH_SIZE)
    test_dataloader = DummyDataloader(length=5, batch_size=gpc.config.BATCH_SIZE)

    # create loss function
    criterion = CrossEntropyLoss(label_smoothing=0.1)

    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=gpc.config.LEARNING_RATE, weight_decay=gpc.config.WEIGHT_DECAY)

    # create lr scheduler
    lr_scheduler = CosineAnnealingWarmupLR(optimizer=optimizer,
                                           total_steps=gpc.config.NUM_EPOCHS,
                                           warmup_steps=gpc.config.WARMUP_EPOCHS)

    # initialize
    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(model=model,
                                                                         optimizer=optimizer,
                                                                         criterion=criterion,
                                                                         train_dataloader=train_dataloader,
                                                                         test_dataloader=test_dataloader)

    logger.info("Engine is built", ranks=[0])

    for epoch in range(gpc.config.NUM_EPOCHS):
        # training
        engine.train()
        data_iter = iter(train_dataloader)

        if gpc.get_global_rank() == 0:
            description = 'Epoch {} / {}'.format(epoch, gpc.config.NUM_EPOCHS)
            progress = tqdm(range(len(train_dataloader)), desc=description)
        else:
            progress = range(len(train_dataloader))
        for _ in progress:
            # 查看输入数据
            # data, label = next(data_iter)
            # print("data, label--->", data.shape, label.shape)
            engine.zero_grad()

            # ---------------------------------------------------------------
            if True:
                # 方式1：支持数据并行+Tensor并行+流水并行
                # 执行 前向、损失计算、反向。 返回(output, label, loss)
                engine.execute_schedule(data_iter, return_output_label=False)
            else:
                # 方案2：数据并行+Tensor并行，不支持流水并行
                imgs, labels = next(data_iter)
                imgs, labels = imgs.cuda(), labels.cuda()
                output = engine(imgs)
                loss = engine.criterion(output, labels)
                engine.backward(loss)
            # ---------------------------------------------------------------
            engine.step()
            lr_scheduler.step()
    gpc.destroy()


if __name__ == '__main__':
    main()

# 启动命令  4卡=流水并行2 * Tensor并行2
# colossalai run --nproc_per_node 4 train.py --config config.py