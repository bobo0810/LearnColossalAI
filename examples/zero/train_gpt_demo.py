import os
from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from commons.model_zoo import model_builder
from commons.utils import get_data, get_profile_context, get_tflops, get_time_stamp
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

CAI_VERSION = colossalai.__version__

"""
参考https://github.com/hpcaitech/ColossalAI/blob/v0.2.5/examples/language/gpt/gemini/train_gpt_demo.py
"""

def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--distplan",
        type=str,
        default='CAI_Gemini',
        help="The distributed plan [colossalai, zero1, zero2, torch_ddp, torch_zero].",
    ) # 分布式策略
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor Parallelism Degree. Valid when using colossalai as dist plan.",
    ) # Tensor并行度(所需GPU数)   1d=1  2d=2  2.5d=4  3d=8
    parser.add_argument(
        "--placement",
        type=str,
        default='cpu',
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
    ) # Gemini的模型数据（模型参数、梯度、优化器状态等）放置策略   可选"cpu", "cuda", "auto"
    parser.add_argument(
        "--shardinit",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    ) # 初始化模型时对张量进行分片，以缩小分配设备上的峰值内存大小。
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per DP group of training.",
    ) # 每个数据并行的batch
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2_medium",
        help="model model scale",
    ) # 模型名称
    parser.add_argument(
        "--train_step",
        type=int,
        default=10,
        help="training iterations for test",
    ) #训练迭代数

    args = parser.parse_args()
    return args


# Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split('\n')[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


# Tensor Parallel
# Tensor Parallel
def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    """tensor_parallelize   Tensor并行，将模型参数初始化到各进程组
    Sharding the Model Parameters. 共享模型参数

    Args:
        model (torch.nn.Module): a torch module to be sharded
    """
    # 遍历整个模型
    for mn, module in model.named_modules():
        # 遍历每一层的参数及权重
        for pn, param in module.named_parameters(recurse=False):
            # NOTE() a param maybe shared by two modules  一个参数可能被两个模块共享
            # 当参数存在"visited"属性，说明已被访问过。故跳过。
            if hasattr(param, 'visited'):
                continue

            # if shard init, then convert param to replica and use the dp-only ProcessGroup
            # 如果开启tensor并行，则将param转为副本并分到进程组
            param: ColoParameter = param  # param属于ColoParameter类型
            param.set_dist_spec(ReplicaSpec())
            param.set_process_group(pg) # 仅适用于数据并行和Tensor并行

            # shard it w.r.t tp pattern   将GPT TransformerBlock中的MLP层 划分为tensor并行
            if 'mlp.c_fc' in mn: # fc层（实际是Conv1D）的权重和偏置 根据传入的进程组，tensor1d并行
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)    # colmn slice 按列划分
                    # keep the shape of the output from c_fc 保持输出形状
                    param.compute_spec.set_output_replicate(False)
                else:
                    # 张量并行进程中 复制张量
                    param.set_dist_spec(ReplicaSpec())
            elif 'mlp.c_proj' in mn: # 实际也是Conv1D
                if 'weight' in pn:
                    split_param_row_tp1d(param, pg)    # row slice 按行划分
                else:
                    param.set_dist_spec(ReplicaSpec()) # 复制张量

            # Transformer的线性投射层  wte：图像编码   wpe:位置编码position embedding
            # nn.Embedding类型，是一个简单的存储固定大小的词典的嵌入向量的查找表。即给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系。输入为一个编号列表，输出为对应的符号嵌入向量列表。
            elif 'wte' in mn or 'wpe' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice按列划分

            # attention层 （c_attn、c_proj实际是Conv1D）
            elif 'c_attn' in mn or 'c_proj' in mn:
                split_param_col_tp1d(param, pg)    # colmn slice按列划分
            else:
                param.set_dist_spec(ReplicaSpec())  # 复制张量
            param.visited = True


def main():
    # version check
    # this example is supposed to work for versions greater than 0.2.0
    assert version.parse(CAI_VERSION) >= version.parse("0.2.0")

    set_cpu_maximum_parallelism()# 设置CPU最大并行数
    args = parse_args()

    # if args.distplan not in ["colossalai", "torch_ddp", "torch_zero", "zero1", "zero2"]:
    if args.distplan not in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]:
        raise TypeError(f"{args.distplan} is error")

    # batch size per DP degree
    BATCH_SIZE = args.batch_size # 每个数据并行的batch
    SEQ_LEN = 1024
    VOCAB_SIZE = 50257

    NUM_STEPS = args.train_step # 训练迭代总数

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median"
    PROF_FLAG = False    # The flag of profiling, False by default

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f"{args.model_type}, {args.distplan}, batch size {BATCH_SIZE}", ranks=[0])

    # build criterion 初始化损失函数
    criterion = GPTLMLoss()

    torch.manual_seed(123)
    if args.distplan.startswith("CAI"): # 采用colossalai训练框架
        # all param must use the same process group. 所有参数必须在同一进程组
        world_size = torch.distributed.get_world_size() # 获取当前进程组的进程总数
        shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None  # 如果初始化时开启tensor分片，则创建进程组，否则为None
        default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None # 如果初始化时开启tensor分片，则创建分布式规范   否则为None

        if args.shardinit and args.distplan != "CAI_Gemini":
            raise RuntimeError("You can only use shardinit with CAI_Gemini")

        # build GPT model 初始化模型
        with ColoInitContext(device=get_current_device(), #初始化到指定设备
                             dtype=torch.half, # 数据类型
                             default_dist_spec=default_dist_spec,
                             default_pg=shard_pg):
            model = model_builder(args.model_type)(checkpoint=True)  # huggingface的普通模型

        # 创建进程组
        tp_pg = ProcessGroup(tp_degree=args.tp_degree)
        # Tensor Parallelism (TP)
        # You should notice that v0.1.10 is not compatible with TP degree > 1
        if args.tp_degree > 1:
            tensor_parallelize(model, tp_pg) # Tensor并行，将模型参数初始化到各进程组

        # asign running configurations运行配置
        gemini_config = None
        if args.distplan.startswith("CAI_ZeRO"): # 启用ZeRO
            optim_config = dict(reduce_bucket_size=12 * 1024 * 1024, overlap_communication=True, verbose=True)
        elif args.distplan == "CAI_Gemini": # 启用ZeRO优化器和Gmeini
            gemini_config = dict(strict_ddp_mode=args.tp_degree == 1, # 若tensor并行为1，则为严格ddp模式
                                 device=get_current_device(),
                                 placement_policy=args.placement,
                                 pin_memory=True,
                                 hidden_dim=model.config.n_embd,
                                 search_range_mb=128)
            # 仅在使用hybrid CPU optimizer，且 placement_policy='auto' 时有效
            optim_config = dict(gpu_margin_mem_ratio=0.)
        else:
            raise RuntimeError

        # build a highly optimized gpu/cpu optimizer 初始化 高度优化的优化器，支持优化器状态卸载到NVMe
        optimizer = HybridAdam(model.parameters(), lr=1e-3)

        if args.distplan == "CAI_ZeRO1":
            zero_stage = 1
        elif args.distplan == "CAI_ZeRO2":
            zero_stage = 2
        elif args.distplan == "CAI_Gemini":
            zero_stage = 3
        else:
            raise RuntimeError

        # wrap your model and optimizer
        model = zero_model_wrapper(model, zero_stage, gemini_config) # 模型包装为zero_model
        optimizer = zero_optim_wrapper(model, optimizer, optim_config=optim_config)# 优化器包装为zero_optim

        logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])
    elif args.distplan.startswith("Pytorch"):
        assert args.tp_degree == 1, "The degree of TP should be 1 for DDP examples."
        model = model_builder(args.model_type)(checkpoint=True).cuda()
        model = DDP(model)
        if args.distplan.endswith("DDP"):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif args.distplan.endswith("ZeRO"):
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=1e-3)
    else:
        raise RuntimeError

    # model is shared after TP
    numel = get_model_size(model)
    logger.info(f"the size of testing model size is {model_size_formatter(numel)}.")
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, SEQ_LEN)

    torch.cuda.synchronize()  # 等待当前设备上所有流中的所有核心完成
    model.train()
    tflops_list = []

    def train_step():
        # we just use randomly generated data here
        input_ids, attn_mask = get_data(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        optimizer.zero_grad()

        start = time()
        outputs = model(input_ids, attn_mask)
        loss = criterion(outputs, input_ids)
        torch.cuda.synchronize()
        fwd_end = time()
        fwd_time = fwd_end - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])

        if args.distplan.startswith("CAI"):
            optimizer.backward(loss)
        elif args.distplan.startswith("Pytorch"):
            loss.backward()
        else:
            raise RuntimeError

        torch.cuda.synchronize()
        bwd_end = time()
        bwd_time = bwd_end - fwd_end
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])

        optimizer.step()
        torch.cuda.synchronize()
        optim_time = time() - bwd_end
        step_time = time() - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Optimizer step '), ranks=[0])

        step_tflops = get_tflops_func(step_time)
        logger.info(
            f"[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s",
            ranks=[0],
        )
        if n >= WARMUP_STEPS:
            tflops_list.append(step_tflops)

    demo_profiler = get_profile_context(PROF_FLAG,
                                        WARMUP_STEPS,
                                        NUM_STEPS - WARMUP_STEPS,
                                        save_dir=f"profile/{get_time_stamp()}-demo")

    with demo_profiler as prof:
        for n in range(NUM_STEPS):
            train_step()
            prof.step()

    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    logger.info(f"Median TFLOPS is {tflops_list[median_index]:.3f}")
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
