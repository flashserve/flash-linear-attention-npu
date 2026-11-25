# Qwen GDN A5 模型形状回归

完整 A5 NPU CI 在安装本次待测的 adapter 和 OPP 后运行这组测试。复测可在同一环境、仓库根目录执行：

```sh
python3 ci/gdn_a5/run.py --device 0 --report .ci-tmp/gdn-a5-model.json
```

三条 Prefill 用例传入模型原始 Q/K/V 头数，与 CPU 递推参考逐元素对比，同时从设备 trace 检查 Prepare、H、O 三个 kernel 的执行顺序。两条 Decode 用例采用单步请求和互不重叠的状态槽，对比输出和全部状态回写。具体形状见同目录的 `cases.json`。测试输入使用固定种子，覆盖形状与算子语义；整网权重和线上输入可另行验证。

输出和状态采用仓库 GDN Example/ST 的输出阈值，绝对和相对容差均为 0.005，余弦相似度至少为 0.999。任一用例超差或路径检查失败即返回非零。逐用例结果在 NPU CI 的 A5 运行产物中归档。
