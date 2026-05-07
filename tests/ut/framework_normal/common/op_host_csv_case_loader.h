/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_HOST_CSV_CASE_LOADER_H
#define OP_HOST_CSV_CASE_LOADER_H

#include "csv_case_load_utils.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"

const gert::TilingContextPara::TensorDescription TD_DEFAULT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};
const gert::InfershapeContextPara::TensorDescription ID_DEFAULT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};

const std::unordered_map<std::string, ge::DataType> GE_DTYPE {
    {"FLOAT", ge::DT_FLOAT},
    {"FLOAT16", ge::DT_FLOAT16},
    {"INT8", ge::DT_INT8},
    {"INT16", ge::DT_INT16},
    {"UINT16", ge::DT_UINT16},
    {"UINT8", ge::DT_UINT8},
    {"INT32", ge::DT_INT32},
    {"INT64", ge::DT_INT64},
    {"UINT32", ge::DT_UINT32},
    {"UINT64", ge::DT_UINT64},
    {"BOOL", ge::DT_BOOL},
    {"DOUBLE", ge::DT_DOUBLE},
    {"STRING", ge::DT_STRING},
    {"DUAL_SUB_INT8", ge::DT_DUAL_SUB_INT8},
    {"DUAL_SUB_UINT8", ge::DT_DUAL_SUB_UINT8},
    {"COMPLEX64", ge::DT_COMPLEX64},
    {"COMPLEX128", ge::DT_COMPLEX128},
    {"QINT8", ge::DT_QINT8},
    {"QINT16", ge::DT_QINT16},
    {"QINT32", ge::DT_QINT32},
    {"QUINT8", ge::DT_QUINT8},
    {"QUINT16", ge::DT_QUINT16},
    {"RESOURCE", ge::DT_RESOURCE},
    {"STRING_REF", ge::DT_STRING_REF},
    {"DUAL", ge::DT_DUAL},
    {"VARIANT", ge::DT_VARIANT},
    {"BF16", ge::DT_BF16},
    {"UNDEFINED", ge::DT_UNDEFINED},
    {"INT4", ge::DT_INT4},
    {"UINT1", ge::DT_UINT1},
    {"INT2", ge::DT_INT2},
    {"UINT2", ge::DT_UINT2},
    {"COMPLEX32", ge::DT_COMPLEX32},
    {"HIFLOAT8", ge::DT_HIFLOAT8},
    {"FLOAT8_E5M2", ge::DT_FLOAT8_E5M2},
    {"FLOAT8_E4M3FN", ge::DT_FLOAT8_E4M3FN},
    {"FLOAT8_E8M0", ge::DT_FLOAT8_E8M0},
    {"FLOAT6_E3M2", ge::DT_FLOAT6_E3M2},
    {"FLOAT6_E2M3", ge::DT_FLOAT6_E2M3},
    {"FLOAT4_E2M1", ge::DT_FLOAT4_E2M1},
    {"FLOAT4_E1M2", ge::DT_FLOAT4_E1M2}
};

const std::unordered_map<std::string, ge::Format> GE_FORMAT {
    {"NCHW", ge::FORMAT_NCHW},
    {"NHWC", ge::FORMAT_NHWC},
    {"ND", ge::FORMAT_ND},
    {"NC1HWC0", ge::FORMAT_NC1HWC0},
    {"FRACTAL_Z", ge::FORMAT_FRACTAL_Z},
    {"NC1C0HWPAD", ge::FORMAT_NC1C0HWPAD},
    {"NHWC1C0", ge::FORMAT_NHWC1C0},
    {"FSR_NCHW", ge::FORMAT_FSR_NCHW},
    {"FRACTAL_DECONV", ge::FORMAT_FRACTAL_DECONV},
    {"C1HWNC0", ge::FORMAT_C1HWNC0},
    {"FRACTAL_DECONV_TRANSPOSE", ge::FORMAT_FRACTAL_DECONV_TRANSPOSE},
    {"FRACTAL_DECONV_SP_STRIDE_TRANS", ge::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
    {"NC1HWC0_C04", ge::FORMAT_NC1HWC0_C04},
    {"FRACTAL_Z_C04", ge::FORMAT_FRACTAL_Z_C04},
    {"CHWN", ge::FORMAT_CHWN},
    {"FRACTAL_DECONV_SP_STRIDE8_TRANS", ge::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
    {"HWCN", ge::FORMAT_HWCN},
    {"NC1KHKWHWC0", ge::FORMAT_NC1KHKWHWC0},
    {"BN_WEIGHT", ge::FORMAT_BN_WEIGHT},
    {"FILTER_HWCK", ge::FORMAT_FILTER_HWCK},
    {"HASHTABLE_LOOKUP_LOOKUPS", ge::FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
    {"HASHTABLE_LOOKUP_KEYS", ge::FORMAT_HASHTABLE_LOOKUP_KEYS},
    {"HASHTABLE_LOOKUP_VALUE", ge::FORMAT_HASHTABLE_LOOKUP_VALUE},
    {"HASHTABLE_LOOKUP_OUTPUT", ge::FORMAT_HASHTABLE_LOOKUP_OUTPUT},
    {"HASHTABLE_LOOKUP_HITS", ge::FORMAT_HASHTABLE_LOOKUP_HITS},
    {"C1HWNCoC0", ge::FORMAT_C1HWNCoC0},
    {"MD", ge::FORMAT_MD},
    {"NDHWC", ge::FORMAT_NDHWC},
    {"FRACTAL_ZZ", ge::FORMAT_FRACTAL_ZZ},
    {"FRACTAL_NZ", ge::FORMAT_FRACTAL_NZ},
    {"NCDHW", ge::FORMAT_NCDHW},
    {"DHWCN", ge::FORMAT_DHWCN},
    {"NDC1HWC0", ge::FORMAT_NDC1HWC0},
    {"FRACTAL_Z_3D", ge::FORMAT_FRACTAL_Z_3D},
    {"CN", ge::FORMAT_CN},
    {"NC", ge::FORMAT_NC},
    {"DHWNC", ge::FORMAT_DHWNC},
    {"FRACTAL_Z_3D_TRANSPOSE", ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE},
    {"FRACTAL_ZN_LSTM", ge::FORMAT_FRACTAL_ZN_LSTM},
    {"FRACTAL_Z_G", ge::FORMAT_FRACTAL_Z_G},
    {"RESERVED", ge::FORMAT_RESERVED},
    {"ALL", ge::FORMAT_ALL},
    {"NULL", ge::FORMAT_NULL},
    {"ND_RNN_BIAS", ge::FORMAT_ND_RNN_BIAS},
    {"FRACTAL_ZN_RNN", ge::FORMAT_FRACTAL_ZN_RNN},
    {"NYUV", ge::FORMAT_NYUV},
    {"NYUV_A", ge::FORMAT_NYUV_A},
    {"NCL", ge::FORMAT_NCL}
};

inline ge::DataType Str2DTypeGE(const std::string& dtypeStr)
{
    return ReadMap(GE_DTYPE, dtypeStr, ge::DT_UNDEFINED);
}

inline ge::graphStatus Str2StatusGE(const std::string& statusStr)
{
    return statusStr == "SUCCESS" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

inline gert::StorageShape GetStorageShape(const std::string& shapeArrStr)
{
    gert::StorageShape shape;
    std::vector<int64_t> shapeArr = GetShapeArr(shapeArrStr);
    switch (shapeArr.size()) {
        case 0:
            break;
        case 1:
            shape = gert::StorageShape({shapeArr[0]}, {shapeArr[0]});
            break;
        case 2:
            shape = gert::StorageShape({shapeArr[0], shapeArr[1]}, {shapeArr[0], shapeArr[1]});
            break;
        case 3:
            shape = gert::StorageShape({shapeArr[0], shapeArr[1], shapeArr[2]},
                {shapeArr[0], shapeArr[1], shapeArr[2]});
            break;
        case 4:
            shape = gert::StorageShape({shapeArr[0], shapeArr[1], shapeArr[2], shapeArr[3]},
                {shapeArr[0], shapeArr[1], shapeArr[2], shapeArr[3]});
            break;
        case 5:
            shape = gert::StorageShape({shapeArr[0], shapeArr[1], shapeArr[2], shapeArr[3], shapeArr[4]},
                {shapeArr[0], shapeArr[1], shapeArr[2], shapeArr[3], shapeArr[4]});
            break;
        default:
            std::cout << "[ERROR] Shape " << shapeArr.size() << " not support!" << std::endl;
            break;
    }
    return shape;
}

template<typename T>
inline int GetTensorGE(const csv_map& csvMap, const std::string& shapeKey, const std::string& dtypeKey,
    const std::string& formatKey, T& out)
{
    std::string shapeStr = ReadMap(csvMap, shapeKey);
    std::string dtypeStr = ReadMap(csvMap, dtypeKey);
    std::string formatStr = ReadMap(csvMap, formatKey);
    if (shapeStr.empty() && dtypeStr.empty() && formatStr.empty()) {
        return 0;
    }

    gert::StorageShape shape = GetStorageShape(shapeStr);
    ge::DataType dtype = ReadMap(GE_DTYPE, dtypeStr, ge::DT_UNDEFINED);
    ge::Format format = ReadMap(GE_FORMAT, formatStr, ge::FORMAT_NULL);
    out = T(shape, dtype, format);
    return 1;
}

inline int GetDataTypeGE(const csv_map& csvMap, const std::string& dtypeKey, ge::DataType& out)
{
    std::string dtypeStr = ReadMap(csvMap, dtypeKey);
    if (dtypeStr.empty()) return 0;

    out = Str2DTypeGE(dtypeStr);
    return 1;
}

struct HostUtParamBase {
    std::string case_name;
    ge::graphStatus expectResult;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;

    HostUtParamBase(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->expectResult = Str2StatusGE(ReadMap(csvMap, "expectResult"));
    }
};

inline std::ostream& operator<<(std::ostream& os, const HostUtParamBase& param)
{
    return os << param.case_name;
}

#endif // OP_HOST_CSV_CASE_LOADER_H
