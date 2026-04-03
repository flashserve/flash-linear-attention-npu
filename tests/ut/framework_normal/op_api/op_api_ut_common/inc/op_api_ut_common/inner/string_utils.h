/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef _OP_API_UT_COMMON_STRING_UTILS_H_
#define _OP_API_UT_COMMON_STRING_UTILS_H_

#include <string>
#include <vector>

using namespace std;

vector<string> Split(const string &s, const string& spliter);
string &Ltrim(string &s);
string &Rtrim(string &s);
string Trim(string &s);

#endif