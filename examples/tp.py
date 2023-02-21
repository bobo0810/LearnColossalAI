import colossalai
import colossalai.nn as col_nn
import torch
from colossalai.utils import print_rank_0
from colossalai.utils import get_current_device

"""
参考 https://colossalai.org/zh-Hans/docs/features/1D_tensor_parallel
"""

# --------------------------初始化模型--------------------------------------
class MLP(torch.nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        intermediate_dim = dim * 4
        self.dense_1 = col_nn.Linear(dim, intermediate_dim) # colossalai的分布式算子，以支持tensor并行
        print_rank_0(
            f"Weight of the first linear layer: {self.dense_1.weight.transpose(0, 1).shape}"
        )
        self.activation = torch.nn.GELU()
        self.dense_2 = col_nn.Linear(intermediate_dim, dim)
        print_rank_0(
            f"Weight of the second linear layer: {self.dense_2.weight.transpose(0, 1).shape}"
        )
        self.dropout = col_nn.Dropout(0.1)

    def forward(self, x):
        x = self.dense_1(x)
        print_rank_0(f"Output of the first linear layer: {x.shape}")
        x = self.activation(x)
        x = self.dense_2(x)
        print_rank_0(f"Output of the second linear layer: {x.shape}")
        x = self.dropout(x)
        return x
m = MLP()  # 初始化模型
# --------------------------初始化配置--------------------------------------
# 参数配置
CONFIG = dict(
    parallel=dict(
        data=1,
        pipeline=1,
        tensor=dict(size=2, mode="1d"), # tensor 1D并行
    )
)
colossalai.launch_from_torch(
    config=CONFIG,
)


# --------------------------调用模型--------------------------------------
x = torch.randn((16, 256), device=get_current_device()) # 当前进程生成完整Tensor
torch.distributed.broadcast(x, src=0)  # synchronize input  将当前tensor广播到所有进程上
x = m(x) # 前向


# 启动命令
# colossalai run --nproc_per_node 2 tp.py
