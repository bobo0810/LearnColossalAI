# ZeRO + Gemini



两种写法，功能等价

1. 参考代码  https://github.com/hpcaitech/ColossalAI/blob/v0.2.5/examples/language/gpt/gemini/train_gpt_demo.py 

2. 官方文档  https://github.com/hpcaitech/ColossalAI/blob/v0.2.5/docs/source/zh-Hans/features/zero_with_chunk.md

```
model = zero_model_wrapper(model, zero_stage=3, gemini_config=xx)
等价于
model = ZeroDDP(model, gemini_manager)
```

```
optimizer = zero_optim_wrapper(model, optimizer=HybridAdam, optim_config)
等价于
optimizer = GeminiAdamOptimizer(model)
```

