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

### Gemini异构内存空间管理器

<img src="assets/5251677140103_.pic.jpg" width = "800"   align=center />





## 总结

- GPU总数量= 数据并行大小 × 张量并行大小 × 流水并行大小，三者可同时应用
- 指定 张量并行、流水并行的大小，则自动推断数据并行大小
- Engine： 对模型、优化器、损失函数的封装类。
- fp16与ZeRO配置不兼容

### TODO

- ColoTensor ：Pytorch Tensor子类，全局tensor，串行编写，分布训练

