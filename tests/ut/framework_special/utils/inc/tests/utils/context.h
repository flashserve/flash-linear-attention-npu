/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file context.h
 * \brief 提供 CPU模式 Tiling / Kernel 阶段上下文功能, 辅助 Tiling / Kernel 运行.
 */

#pragma once

#include <any>
#include <functional>
#include "tests/utils/context_intf.h"

namespace ops::adv::tests::utils {

class Context : public ops::adv::tests::utils::ContextIntf {
public:
    /**
     * Ascend C 框架推荐的 TilingData 最大长度, 超过此值可能会导致算子性能下降.
     * 注意: 若算子实际 TilingData 长度超过此值, 不可直接修改此值, 而应调用 SetTilingDataMaxSize 接口修改.
     */
    static constexpr uint32_t kDefaultTilingDataMaxSize = 2048;

    /**
     * Kernel 运行回调函数
     *
     * \param func 算子 kernel 入口函数
     * \param tilingKey TilingKey
     * \param blockDim Tiling 切分 BlockDim
     * \param inputs 算子输入
     * \param outputs 算子输出
     * \param workspace 运行所需的 workspace 空间
     * \param tilingData TilingData 结果
     */
    typedef bool (*KernelRunCbf)(void *func, uint64_t tilingKey, int64_t blockDim, std::vector<TensorIntf *> &inputs,
                                 std::vector<TensorIntf *> &outputs, uint8_t *workspace, uint8_t *tilingData);

    Context() = default;
    ~Context() override;

    /**
     * 属性设置
     * 设置确定性计算标识
     */
    [[maybe_unused]] [[nodiscard]] bool SetDeterministic(bool enable);

    /**
     * 属性设置
     * 设置算子输入属性
     */
    [[maybe_unused]] [[nodiscard]] bool SetAttrs(std::vector<std::pair<std::string, std::any>> attrs);

    /**
     * 属性设置
     * 添加算子输入属性
     */
    [[maybe_unused]] [[nodiscard]] bool AddAttrs(std::vector<std::pair<std::string, std::any>> attrs);

    /**
     * 属性设置
     * 修改算子输入属性
     */
    [[maybe_unused]] [[nodiscard]] bool MdfAttrs(const std::pair<std::string, std::any> &attr);

    /**
     * 属性设置
     * 设置算子 TilingData 最大大小
     */
    [[maybe_unused]] [[nodiscard]] bool SetTilingDataMaxSize(uint32_t size = kDefaultTilingDataMaxSize);

    /**
     * 属性设置
     * 设置算子 Kernel 处理回调函数
     */
    [[maybe_unused]] [[nodiscard]] bool SetKernelRunCbf(KernelRunCbf cbf);

    /**
     * 属性设置
     * 设置算子 Kernel 总入口回调函数
     */
    [[maybe_unused]] [[nodiscard]] bool SetKernelMainFunc(void *func);

    /**
     * 属性获取
     * 获取 TilingData 数量
     */
    [[maybe_unused]] [[nodiscard]] int32_t GetTilingDataNum() const;

    /**
     * 属性获取
     * 获取 TilingData 值(无法获取具体结构, 需要调用者进行数据类型转换)
     */
    [[maybe_unused]] [[nodiscard]] const void *GetTilingData() const;

    /**
     * 属性获取
     * 获取 TilingData 的 string 表示(0x格式)
     */
    [[maybe_unused]] [[nodiscard]] const std::string &GetTilingDataStr() const;

    /**
     * 属性获取
     * 获取 json 表示的 Tiling 结果
     */
    [[maybe_unused]] [[nodiscard]] const std::string &GetTilingResult() const;

    /**
     * 运行 Tiling
     * \param caseName 当前运行的用例名
     */
    [[maybe_unused]] [[nodiscard]] bool RunTiling(std::string &caseName, bool withSocversion) override;

protected:
    static constexpr size_t kDefaultTilingResultSize_ = 10 * 1024 * 1024;

    /* 属性 */
    std::vector<std::pair<std::string, std::any>> attrs_;
    uint32_t tilingDataMaxLen_ = kDefaultTilingDataMaxSize;
    KernelRunCbf kernelRunCbf_ = nullptr;
    void *kernelMainFunc_ = nullptr;
    int64_t deterministic_ = 0; /**< 默认不开启确定性计算 */

    /* Tiling */
    std::string inputsJson_;
    std::string outputsJson_;
    std::string attrsJson_;
    std::string extraInfoJson_;
    int32_t tilingDataNum_ = 0;
    bool clearAtomic_ = false;
    std::vector<uint8_t> tilingData_; /**< 向 Kernel 下发的 struct 表示的 TilingData  */
    std::string tilingResult_;        /**< Context 直接返回的 json 表示的 Tiling 结果 */
    std::string tilingDataStr_;

protected:
    bool RunKernelProcess(std::string &caseName) override;
    uint8_t *AllocWorkspaceImpl(uint64_t size) override;
    void FreeWorkspaceImpl(uint8_t *ptr) override;

    bool InitTilingJsonStr();
    bool ParseTilingResult();

    typedef bool (*CheckKernelResultCbf)(std::string &kernelLog);
    bool CheckKernelResult(bool &ret, std::string &caseName, std::string &path, const char *type,
                           CheckKernelResultCbf cbf, bool detail = true);
    static bool CheckModelKernelResultStr(std::string &kernelLog);
    static bool CheckFrameworkKernelResultStr(std::string &kernelLog);

    template <class T> bool DetectField(T &field, const char *fPrefix, const char *fSuffix)
    {
        char *bgn = nullptr;
        char *end = nullptr;
        if (!this->DetectPosit(&bgn, &end, fPrefix, fSuffix)) {
            return false;
        }
        std::string pre(fPrefix);
        std::string sub(bgn + pre.length(), end - bgn - pre.length());
        Context::DetectValue(sub, field);
        return true;
    }

    bool DetectPosit(char **bgn, char **end, const char *fPrefix, const char *fSuffix);
    [[maybe_unused]] static void DetectValue(std::string &sub, int64_t &value);
    [[maybe_unused]] static void DetectValue(std::string &sub, uint64_t &value);
    [[maybe_unused]] static void DetectValue(std::string &sub, bool &value);
    [[maybe_unused]] static void DetectValue(std::string &sub, std::string &value);

private:
    void Destroy();
};

} // namespace ops::adv::tests::utils
