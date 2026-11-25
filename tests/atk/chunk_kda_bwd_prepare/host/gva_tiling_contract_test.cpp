#include <cassert>
#include <iostream>
#include "tiling_base/tiling_templates_registry.h"
#undef OP_LOGD
#undef OP_LOGE
#define OP_LOGD(...) ((void)0)
#define OP_LOGE(...) ((void)0)
#include "fla/ops/ascendc/kda/chunk_kda_bwd_prepare/op_host/op_tiling/chunk_kda_bwd_prepare_tiling_processor.h"
int main() {
 for (bool variable: {false,true}) for (int64_t ratio=1;ratio<=16;++ratio) {
  const int64_t hv=ratio*3;
  gert::StorageShape a({1,hv,65,64},{1,hv,65,64}),v({1,hv,65,128},{1,hv,65,128});
  gert::StorageShape h({1,2,hv,128,128},{1,2,hv,128,128}),cu({3},{3}),indices({4},{4});
  KDA::ChunkKdaBwdPrepareTilingData data{};
  optiling::ChunkKdaBwdPrepareTilingContext ctx{"gva",&a,&v,&v,&h,variable?&cu:nullptr,variable?&indices:nullptr,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,.125,64,false,"BNSD",24,4096,false,false,false,ratio};
  optiling::ChunkKdaBwdPrepareTilingProcessor tiler(ctx,data);
  assert(tiler.Process()==ge::GRAPH_SUCCESS);
  assert(data.HK==3 && data.NV==hv && data.gvaRatio==ratio);
  assert(tiler.GetTilingKey()==GET_TPL_TILING_KEY(KDA::BNSD,variable,0,ratio>1));
  assert(GET_TPL_TILING_KEY(KDA::BNSD,variable,0,0)!=GET_TPL_TILING_KEY(KDA::BNSD,variable,0,1));
  std::cout<<"ratio="<<ratio<<" variable="<<variable<<" key="<<tiler.GetTilingKey()<<'\n';
  for (int64_t bad: {int64_t(0),int64_t(17),hv+1}) {ctx.gvaRatio=bad;assert(tiler.Process()==ge::GRAPH_FAILED);}
 }
}
