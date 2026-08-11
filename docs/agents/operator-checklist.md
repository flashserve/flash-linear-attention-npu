# 算子开发交付清单

新增或修改 Ascend C 算子时，按改动范围逐项确认。未涉及的项目应明确标记为不适用，而不是直接忽略。

## 接口与实现

- [ ] `op_host/*_def.cpp` 的输入、输出、属性、dtype、format 和 required/optional 定义与既有契约一致。
- [ ] InferShape、参数校验、TilingData 和 TilingKey 已同步。
- [ ] aclnn op_api 的签名、workspace 查询、执行器和返回语义已同步。
- [ ] kernel 的模板、数据布局、同步和写回语义与 host tiling 一致。
- [ ] schema、生成输入、稳定 Python 导出和 ctypes ABI 已同步。
- [ ] 公开参数名称、数量、顺序、默认值和行为保持兼容；任何例外已事前取得明确确认。
- [ ] 所有支持 SOC 复用同一个 L0 定义、原型和 L2 调用路径。

## 算法与优化

- [ ] 数学语义、初始/最终状态和有效输出区域已经固定。
- [ ] 已判断 chunk 间是否存在 carry，并阅读对应优化指南。
- [ ] producer-consumer DAG、并行轴、stage 边界和中间量落点已经记录。
- [ ] workspace segment、slot、dtype、layout、ready 和 free 生命周期完整。
- [ ] 性能目标、功能范围和模板优势域分别声明，没有用 shape 特例缩窄功能范围。
- [ ] 4-head window 默认方案或受控例外已经完成容量、同步和性能论证。
- [ ] L1/L0/UB resident、双缓冲、直连通路和缓存都按最后消费点释放。
- [ ] 优化没有引入第二套 L0、冗余 L0 参数或长期 fallback 路径。

## 测试与文档

- [ ] 单算子测试包含参考实现、正向场景和必要的反向拦截。
- [ ] fixed/varlen、dtype、layout、head ratio、完整/尾 chunk、完整/尾 head window 已按风险覆盖。
- [ ] 修改公共组件、ABI 或跨平台路径时已扩大回归范围。
- [ ] 精度失败没有通过缩小 range、跳过 case 或放宽阈值规避。
- [ ] 性能结论来自固定基线和 profiling，不以 Python wall time 代替。
- [ ] 对疑似 race、越界、未初始化或同步问题执行了对应检查。
- [ ] 算子 README、aclnn 文档、示例和 CI case 与代码同步。

## 提交前

- [ ] `git status --short` 只包含本次任务文件。
- [ ] `git diff --check` 通过。
- [ ] 没有构建产物、缓存、profile 输出、临时数据或日志进入提交。
- [ ] 公开说明不包含内部机器、账号、绝对路径、临时目录或环境信息。
- [ ] 已执行和未执行的验证均有明确说明。
