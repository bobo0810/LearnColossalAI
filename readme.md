# 一起来训大模型

- 该仓库收录于[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)
- [ColossalAI官网](https://www.colossalai.org/zh-Hans/)   [文档](https://www.colossalai.org/zh-Hans/docs/get_started/installation/)

<img src="assets/并行技术.jpg" width = "500"   align=center />


## 最小示例

### Tensor并行
<img src="assets/normal.jpeg" width = "400"   align=center />

- 1D并行  `examples/tp.py`

<img src="assets/1d.jpeg" width = "600"   align=center />



### 流水并行

  `examples/pp.py`

<img src="assets/5241677052951_.pic.jpg" width = "700"   align=center />

### 混合并行

  `examples/hybrid`

### 异构内存空间管理器Gemini

<img src="assets/5251677140103_.pic.jpg" width = "800"   align=center />



### 自动并行AutoParallel(实验性)

`examples/auto`

<img src="assets/68B17411-E4E7-4A22-B8C6-F47F6045FF18.png" width = "400"   align=center />

## 总结

- GPU总数量= 数据并行大小 × 张量并行大小 × 流水并行大小，三者可同时应用
- 指定 张量并行、流水并行的大小，则自动推断数据并行大小
- Engine： 对模型、优化器、损失函数的封装类。
- fp16与ZeRO配置不兼容
- 大batch训练的优化器Lamb, Lars

## 注意

- 配置文件的batch 指单个GPU上的batch。 故总batch 是Nx batch
- 配置文件的epoch 指单个GPU上的epoch。 故总epoch 是Nx epoch

> 每个gpu上的dataloader都是完整的数据集，未做拆分。 故epoch=3  gpu=2时模型实际上过了6遍数据集

### TODO

- ColoTensor ：Pytorch Tensor子类，全局tensor，串行编写，分布训练
- ZeRO



## 实验

| 并行类型   | 并行程度           | 卡1显存占用 | 卡2显存占用 |
| ---------- | ------------------ | ----------- | ----------- |
| 数据并行   | size=2             | 8082M       | 8082M       |
| 流水并行   | size=2             | 8252M       | 1974M       |
| Tensor并行 | size=2   mode="1d" | 8079M       | 8079M       |

注：两张卡  ResNet50  Batch=128
