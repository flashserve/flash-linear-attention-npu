// SPDX-License-Identifier: BSD-3-Clause
#include "stable/boxed.h"
#include "stable/vdn.h"
#include "stable/runtime.h"
#include "stable/vdn_manual_buffers.h"
#include <array>
#include <dlfcn.h>

namespace vdn_manual {
constexpr const char* schema = "npu_vdn_pipeline("
    "Tensor q, Tensor k, Tensor v, Tensor beta, "
    "Tensor alpha, Tensor bounds, Tensor text, Tensor k_sp, "
    "Tensor k_tm, Tensor v_sp, Tensor v_tm, Tensor conv_t, "
    "Tensor gram_a_t, Tensor gram_b_t, Tensor(a!) packed, Tensor(a!) filter_buf, "
    "Tensor(a!) conv_out, Tensor(a!) panel, Tensor(a!) k_time, Tensor(a!) v_time, "
    "Tensor(a!) key, Tensor(a!) value, Tensor(a!) query, Tensor(a!) k32, "
    "Tensor(a!) wk32, Tensor(a!) gram_a, Tensor(a!) b16, Tensor(a!) b32, "
    "Tensor(a!) negative, Tensor(a!) first, Tensor(a!) inv, Tensor(a!) eye, "
    "Tensor(a!) two_eye, Tensor(a!) work, Tensor(a!) trans, Tensor(a!) inj, "
    "Tensor(a!) prefix, Tensor(a!) suffix, Tensor(a!) logs_tmp, Tensor(a!) logs, "
    "Tensor(a!) state, Tensor(a!) out, "
    "int frames, int heads, int bridge_panel, int steps, bool conv, bool seed, int stream, int cores=20) -> ()";

using Kernel = void (*)(void**, int, int, int, int, int, int, void*, int);

Kernel load_kernel() {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&load_kernel), &info) || !info.dli_fname)
    throw std::runtime_error("cannot locate the VDN stable launcher");
  std::string path(info.dli_fname);
  path.resize(path.rfind('/') + 1);
  path += "ops/vdn/libvdn_pipeline_device.so";
  void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library) throw std::runtime_error(dlerror());
  auto kernel = reinterpret_cast<Kernel>(dlsym(library, "vdn_forward_pipeline_v2"));
  if (!kernel) throw std::runtime_error(dlerror());
  return kernel;
}

void run(StableIValue* stack, uint64_t inputs, uint64_t outputs) {
  using namespace fla_npu_stable::stable;
  using torch::stable::Tensor;
  if (inputs != BufferCount + 8 || outputs != 0)
    throw std::invalid_argument("invalid VDN pipeline schema");
  const auto f = unbox<int64_t>(stack[BufferCount]), h = unbox<int64_t>(stack[BufferCount+1]);
  const auto panel = unbox<int64_t>(stack[BufferCount+2]), steps = unbox<int64_t>(stack[BufferCount+3]);
  const bool conv = unbox<bool>(stack[BufferCount+4]), seed = unbox<bool>(stack[BufferCount+5]);
  const auto stream = unbox<int64_t>(stack[BufferCount+6]);
  const auto cores = unbox<int64_t>(stack[BufferCount+7]);
  if ((cores!=20 && cores!=24) || !((f==85 && h==7) || (f==87 && (h==7 || h==14))) ||
      panel != (h==7 ? 20 : 44) || steps<1 || steps>15)
    throw std::invalid_argument("invalid VDN pipeline geometry");
  constexpr int64_t dim = 128, square = dim*dim;
  const auto fh = f*h, rows = fh*405, features = rows*dim, matrices = fh*square;
  const auto bank = (f+1)*h*square;
  const std::array<int64_t,BufferCount> counts = {
      features,features,features,rows,fh*dim,f*2,seed?h*square:0,
      conv?h*dim*25:0,conv?h*dim*5:0,conv?h*dim*25:0,conv?h*dim*5:0,
      256+4*(6480+8192),256,256,conv?features:0,conv?h*8*25*256:0,
      conv?features:0,conv?features:0,conv?h*dim*5:0,conv?h*dim*5:0,
      features,features,features,features,features,matrices,matrices,matrices,matrices,
      matrices,matrices,square,square,cores*6*square,matrices,matrices,bank,bank,
      3*fh*dim,(f+1)*h*dim,matrices,features};
  const int bf = aoti_torch_dtype_bfloat16(), fp = aoti_torch_dtype_float32();
  const int u8 = aoti_torch_dtype_uint8(), i64 = aoti_torch_dtype_int64();
  const std::array<int,BufferCount> types = {bf,bf,bf,bf,fp,i64,fp,bf,bf,bf,bf,u8,u8,u8,
      bf,bf,bf,bf,bf,bf,bf,bf,bf,fp,fp,fp,bf,fp,fp,fp,fp,fp,fp,fp,fp,fp,fp,fp,fp,fp,bf,bf};
  auto owners = std::make_shared<std::vector<Tensor>>();
  owners->reserve(BufferCount);
  std::array<void*,BufferCount> ptrs{}, bases{};
  TensorMeta anchor;
  for (int i=0; i<BufferCount; ++i) {
    owners->push_back(unbox<Tensor>(stack[i]));
    if (!counts[i]) continue;
    const auto meta = meta_of(owners->back());
    if (i==Q) anchor = meta;
    fla_npu_stable::vdn::check_same_npu(anchor,meta);
    bool dense = meta.contiguous;
    if (i<=V) dense = meta.sizes==std::vector<int64_t>{f,h,405,dim} &&
                       meta.strides==std::vector<int64_t>{405*h*dim,dim,h*dim,1};
    int64_t numel=1;
    for (auto n:meta.sizes) numel*=n;
    if (!dense || meta.scalar_type!=types[i] || numel!=counts[i] || meta.storage_offset<0 ||
        meta.storage_numel-meta.storage_offset<counts[i])
      throw std::invalid_argument("invalid VDN buffer "+std::to_string(i));
    bases[i]=meta.data;
    ptrs[i]=static_cast<uint8_t*>(meta.data)+meta.storage_offset*element_size(meta.scalar_type);
    if (reinterpret_cast<uintptr_t>(ptrs[i])%32) throw std::invalid_argument("unaligned VDN buffer");
    if (i>=Packed && std::find(bases.begin(),bases.begin()+i,bases[i])!=bases.begin()+i)
      throw std::invalid_argument("overlapping VDN workspace");
  }
  static Kernel kernel = load_kernel();
  auto call = [=]() mutable -> int {
    kernel(ptrs.data(),f,h,panel,steps,conv,seed,reinterpret_cast<void*>(stream),cores);
    owners.reset();
    return 0;
  };
  auto& runtime = fla_npu_stable::Runtime::instance();
  if (runtime.enqueue_enabled()) runtime.enqueue("VdnForwardPipeline",call);
  else call();
}
}  // namespace vdn_manual
