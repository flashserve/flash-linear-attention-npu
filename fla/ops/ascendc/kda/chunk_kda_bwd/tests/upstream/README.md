# fla-org KDA 反向验证

上游固定版本：`c55dc553edf44701bf703715ed052d811f8a6376`。
按用户要求使用 fla-org 测试/标杆，不使用 ATK。安装本分支 wheel 和同次构建 OPP，在 fla-org checkout 根目录运行。

```bash
export ASCEND_RT_VISIBLE_DEVICES=10
export PYTHONPATH=/path/to/flash-linear-attention-npu/fla/ops/ascendc/kda/chunk_kda_bwd/tests/upstream:$PWD
python -m pytest tests/ops/test_kda.py -p public_backward_plugin \
  -k 'test_chunk_collinear_keys_l2norm or test_chunk_varlen_fused_training_contract' -q -s
python -m pytest /path/to/flash-linear-attention-npu/fla/ops/ascendc/kda/chunk_kda_bwd/tests/upstream/test_public_backward_*.py -q -s
```

运行前选择空闲物理设备，不照抄示例设备号。插件只替换受测 chunk_kda 的调用入口，梯度全部来自公开 AscendC ACLNN，未使用 Triton 后备。

- 原始 8 项：原参数、测试体和断言不变，包含既有 BF16/128 回归。
- expanded 20 项：调用相同上游 fused_training_contract 函数，扩展 dtype、D 与保存/重算参数；不宣称是上游原参数集。
- nonsquare 24 项：上游 naive_recurrent_kda 标杆与 assert_close；K64/V32、scale=0.37、空段/尾块、norm 开关、sigmoid 三模式。关闭 sigmoid 时 beta=0，核查 beta 为零仍可有非零 db。
- dense 8 项：相同上游标杆；B=2/T=65、预激活 FP32 gate、有效非零 beta、保存/重算。
- stages 4 项：独立公开小接口，重算缓存逐元素一致；K-first/V-first 物理状态布局；最终梯度对上游标杆。
- invalid 6 项：非法属性/shape/dtype 和超出本轮范围的状态输入，必须由 ACLNN 返回 161002。

组合 h0/dht/dh0 按用户确认不覆盖；没有修改上游状态断言来制造通过结果。详细结果见 ../../docs/validation.md。
