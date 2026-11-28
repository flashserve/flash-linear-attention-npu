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

constexpr const char* feature_schema =
    "npu_vdn_pipeline_from_features(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor(a!) alpha, "
    "Tensor bounds, Tensor(a!) text, Tensor k_sp, Tensor k_tm, Tensor v_sp, Tensor v_tm, "
    "Tensor conv_t, Tensor gram_a_t, Tensor gram_b_t, Tensor(a!) packed, Tensor(a!) filter_buf, "
    "Tensor(a!) conv_out, Tensor(a!) panel, Tensor(a!) k_time, Tensor(a!) v_time, Tensor(a!) key, "
    "Tensor(a!) value, Tensor(a!) query, Tensor(a!) k32, Tensor(a!) wk32, Tensor(a!) gram_a, "
    "Tensor(a!) b16, Tensor(a!) b32, Tensor(a!) negative, Tensor(a!) first, Tensor(a!) inv, "
    "Tensor(a!) eye, Tensor(a!) two_eye, Tensor(a!) work, Tensor(a!) trans, Tensor(a!) inj, "
    "Tensor(a!) prefix, Tensor(a!) suffix, Tensor(a!) logs_tmp, Tensor(a!) logs, Tensor(a!) state, "
    "Tensor(a!) out, int frames, int heads, int bridge_panel, int steps, bool conv, bool seed, "
    "int stream, int cores, Tensor frame_mean, Tensor down_w, Tensor up_w, Tensor dt_bias, "
    "Tensor a_log, Tensor text_k, Tensor text_v, Tensor text_beta, Tensor down_tile, Tensor up_tile, "
    "Tensor text_tile, int width, int length, int text_steps) -> ()";
using FrontKernel = void (*)(void**,void**,int,int,int,int,int,int,int,int,float,void*,int,const int*,int);
using Kernel = void (*)(void**, int, int, int, int, int, int, void*, int);

void* load_kernel(const char* symbol) {
  Dl_info info{};
  if (!dladdr(reinterpret_cast<void*>(&load_kernel), &info) || !info.dli_fname)
    throw std::runtime_error("cannot locate the VDN stable launcher");
  std::string path(info.dli_fname);
  path.resize(path.rfind('/') + 1);
  path += "ops/vdn/libvdn_pipeline_device.so";
  void* library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library) throw std::runtime_error(dlerror());
  auto kernel = dlsym(library, symbol);
  if (!kernel) throw std::runtime_error(dlerror());
  return kernel;
}

template<bool fromFeatures>
void launch(StableIValue* stack, uint64_t inputs, uint64_t outputs) {
  using namespace fla_npu_stable::stable;
  using torch::stable::Tensor;
  if (inputs != BufferCount + 8 + (fromFeatures ? 14 : 0) || outputs != 0)
    throw std::invalid_argument("invalid VDN pipeline schema");
  const auto f = unbox<int64_t>(stack[BufferCount]), h = unbox<int64_t>(stack[BufferCount+1]);
  const auto panel = unbox<int64_t>(stack[BufferCount+2]), steps = unbox<int64_t>(stack[BufferCount+3]);
  const bool conv = unbox<bool>(stack[BufferCount+4]), seed = unbox<bool>(stack[BufferCount+5]);
  const auto stream = unbox<int64_t>(stack[BufferCount+6]);
  const auto cores = unbox<int64_t>(stack[BufferCount+7]);
  if ((cores!=20 && cores!=24) || f<64 || f>96 || h<4 || h>16 ||
      panel<1 || panel>f || steps<1 || steps>15)
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
  owners->reserve(BufferCount + (fromFeatures ? 11 : 0));
  std::array<void*,BufferCount> ptrs{}, bases{};
  TensorMeta anchor;
  for (int i=0; i<BufferCount; ++i) {
    owners->push_back(unbox<Tensor>(stack[i]));
    if (!counts[i]) continue;
    const auto meta = meta_of(owners->back());
    if (i==Inj && meta.sizes==std::vector<int64_t>{0}) continue;
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
  std::array<void*,11> extra{};
  std::array<int,6> strides{};
  int64_t width=0, length=0, iterations=0;
  double scale=0;
  int alphaBf16=0;
  if constexpr(fromFeatures) {
    constexpr int first = BufferCount + 8;
    width=unbox<int64_t>(stack[first+11]);length=unbox<int64_t>(stack[first+12]);
    iterations=unbox<int64_t>(stack[first+13]);scale=2.0 / (2.0 + length);
    if(width<1 || width>65536 || length<0 || length>f*405 || seed!=(length>0)
       || iterations<0 || iterations>24 || !(scale>0 && scale<=1))
      throw std::invalid_argument("invalid VDN frontend geometry");
    std::array<int64_t,11> sizes{f*width,128*width,h*128*128,h*128,h,length*h*128,length*h*128,length*h,256,256,length ? 512 : 0};
    for(int j=0;j<11;++j) {
      owners->push_back(unbox<Tensor>(stack[first+j]));
      const auto m=meta_of(owners->back());
      fla_npu_stable::vdn::check_same_npu(anchor,m);
      int64_t nelem=1;for(auto n:m.sizes) nelem*=n;
      const bool castWeight=j<5 && m.scalar_type==bf;
      if(castWeight) alphaBf16 |= 1<<j;
      const int wanted=j<5?(castWeight?bf:fp):(j<8?bf:u8);
      bool validLayout=m.contiguous;
      int64_t span=nelem;
      if(nelem && j>=5 && j<=7) {
        const auto shape=j==7?std::vector<int64_t>{h,length}:std::vector<int64_t>{length,h,dim};
        if(m.sizes!=shape)throw std::invalid_argument("invalid text layout");
        validLayout=true;span=1;
        for(size_t axis=0;axis<shape.size();++axis) {
          const auto stride=m.strides[axis];
          if(stride<1 || stride>INT32_MAX || (shape[axis]-1)*stride>INT32_MAX)
            throw std::invalid_argument("unsupported text strides");
          span+=(shape[axis]-1)*stride;
        }
        if(span>INT32_MAX || (j!=7 && (m.strides[2]!=1 || m.strides[1]<128 || m.strides[0]<(h-1)*m.strides[1]+128)))
          throw std::invalid_argument("unsupported text span");
        strides[(j-5)*2]=m.strides[0];strides[(j-5)*2+1]=m.strides[1];
      }
      if(!validLayout || m.scalar_type!=wanted || nelem!=sizes[j] || m.storage_offset<0 ||
         m.storage_numel-m.storage_offset<span)
        throw std::invalid_argument("invalid VDN frontend buffer "+std::to_string(j));
      if(!nelem)continue;
      if(m.data==bases[Alpha] || m.data==bases[Seed] ||
         std::find(bases.begin()+Packed,bases.end(),m.data)!=bases.end())
        throw std::invalid_argument("VDN frontend input aliases a writable buffer");
      extra[j]=static_cast<uint8_t*>(m.data)+m.storage_offset*element_size(m.scalar_type);
      if(reinterpret_cast<uintptr_t>(extra[j])%32 && j!=4 && j!=7)
        throw std::invalid_argument("unaligned VDN frontend input");
    }
    for(int j : {Alpha,Seed}) {
      if(!bases[j])continue;
      for(int k=0;k<BufferCount;++k)
        if(k!=j && bases[j]==bases[k])throw std::invalid_argument("overlapping VDN frontend output");
    }
  }

  auto call = [=]() mutable -> int {
    if constexpr(fromFeatures) {
      static const auto kernel=reinterpret_cast<FrontKernel>(load_kernel("vdn_forward_pipeline_from_features"));
      kernel(ptrs.data(),extra.data(),f,h,panel,steps,conv,width,length,iterations,static_cast<float>(scale),reinterpret_cast<void*>(stream),cores,strides.data(),alphaBf16);
    } else {
      static const auto kernel=reinterpret_cast<Kernel>(load_kernel("vdn_forward_pipeline_v2"));
      kernel(ptrs.data(),f,h,panel,steps,conv,seed,reinterpret_cast<void*>(stream),cores);
    }
    owners.reset();
    return 0;
  };
  auto& runtime = fla_npu_stable::Runtime::instance();
  if (runtime.enqueue_enabled()) runtime.enqueue("VdnForwardPipeline",call);
  else call();
}
}  // namespace vdn_manual
