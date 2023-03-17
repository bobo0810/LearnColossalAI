# 大模型食用手册

- 该仓库收录于[PytorchNetHub](https://github.com/bobo0810/PytorchNetHub)
- [ColossalAI官网](https://www.colossalai.org/zh-Hans/)   [文档](https://www.colossalai.org/zh-Hans/docs/get_started/installation/)



## 大模型方案



<img src="assets/训练方案-8869037.svg" width = "1200"   align=center />



## 总结

- GPU总数量= 数据并行大小 × 张量并行大小 × 流水并行大小
- 指定 张量并行、流水并行的大小，则自动推断数据并行大小
- Engine： 对模型、优化器、损失函数的封装类。
- fp16与ZeRO配置不兼容
- Timm库的ViT类模型实现流水并行  参考`examples/images/vit/train.py`

## 注意

- 配置文件的batch 指单个GPU上的batch。 故总batch 是Nx batch
- 配置文件的epoch 指单个GPU上的epoch。 故总epoch 是Nx epoch

> 举例：每个gpu上的dataloader都是完整的数据集，未做拆分。 epoch=3  gpu=2时模型实际上过了6遍数据集

## 实验

| 并行类型   | 等效batch | 并行程度           | 卡1显存占用 | 卡2显存占用 |
| ---------- | --------- | ------------------ |--------|--------|
| 数据并行   | 8*2=16    | size=2             | 3728M  | 3728M  |
| 流水并行   | 16        | size=2             | 4080M  | 1996M  |
| Tensor并行 | 16        | size=2   mode="1d" | 4650M  | 4650M  |

注：Backbone=convnext_base_384_in22ft1k   GPU=2  AMP=true



# 最小示例



<img src="assets/并行技术.jpg" width = "500"   align=center />

### Tensor并行

<img src="assets/normal.jpeg" width = "400"   align=center />

- 1D并行  `examples/tp.py`

<img src="assets/1d.jpeg" width = "600"   align=center />



### 流水并行

  `examples/pp.py`

<img src="assets/5241677052951_.pic.jpg" width = "700"   align=center />

### 混合并行

- 数据并行+流水并行+Tensor并行   `examples/hybrid`

- 问题：显存负载严重不均衡     解决：[流水并行-自定义拆分教程](examples/hybrid/流水并行-自定义拆分.md)

### 零冗余优化器ZeRO

  `examples/zero`

### 异构内存空间管理器Gemini

  `examples/zero`

<img src="assets/5251677140103_.pic.jpg" width = "800"   align=center />



### 自动并行AutoParallel(实验性)

api暂未稳定

<img src="assets/68B17411-E4E7-4A22-B8C6-F47F6045FF18.png" width = "400"   align=center />
