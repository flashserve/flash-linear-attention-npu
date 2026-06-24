#ifndef SOLVE_TRIL_H
#define SOLVE_TRIL_H

#include "kernel_operator.h"
#include "solve_tril_common.h"
#include "mem.h"

using namespace AscendC;


constexpr AscendC::FixpipeConfig CFG_NZ_L1 = {AscendC::CO2Layout::NZ, false};
constexpr AscendC::FixpipeConfig CFG_NZ_UB = {AscendC::CO2Layout::NZ, true};

template <typename InDtype, typename OutDtype>
class SolveTril {
public:
    __aicore__ inline void Init(GM_ADDR aGm, GM_ADDR cu_seqlens, GM_ADDR chunk_indices, GM_ADDR outGm,
                                GM_ADDR workspace, const SolveTrilTilingData *tilingData)
    {
        // Tiling
        batch_size = tilingData->batchSize;
        seq_length = tilingData->seqLength;
        num_head = tilingData->numHead;
        chunk_size = tilingData->chunkSize;
        chunk_num_in_seq = tilingData->chunkNumInSeq;
        chunk_num_total = tilingData->chunkNumTotal;
        mode = tilingData->mode;

        AscendC::printf("wangwei[vector]: batch_size %d\n", batch_size);
        AscendC::printf("wangwei[vector]: seq_length %d\n", seq_length);
        AscendC::printf("wangwei[vector]: num_head %d\n", num_head);
        AscendC::printf("wangwei[vector]: chunk_size %d\n", chunk_size);
        AscendC::printf("wangwei[vector]: chunk_num_in_seq %d\n", chunk_num_in_seq);
        AscendC::printf("wangwei[vector]: chunk_num_total %d\n", chunk_num_total);
        AscendC::printf("wangwei[vector]: mode %d\n", mode);
        // GM
        gm_a.SetGlobalBuffer(reinterpret_cast<__gm__ InDtype *>(aGm));
        gm_cu_seqlens.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cu_seqlens));
        gm_chunk_indices.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(chunk_indices));
        gm_out.SetGlobalBuffer(reinterpret_cast<__gm__ OutDtype *>(outGm));

        // pipe->InitBuffer(ubBuf_, chunk_size * chunk_size * sizeof(InDtype));
        // pipe->InitBuffer(ubBuf_2, chunk_size * chunk_size * sizeof(InDtype));
        // pipe->InitBuffer(l1Buf_, chunk_size * chunk_size * sizeof(InDtype));
        
        // ub_ = ubBuf_.Get<InDtype>();
        // ub_2 = ubBuf_2.Get<InDtype>();
        // l1_ = l1Buf_.Get<InDtype>();


        OnChipBuffer buf;
        
        // UB
        ub_I = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(0);
        ub_Zero = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        ub_A = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 2);
        ub_I_A = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 3);
        ub_Res = buf.template GetBuffer<BufferType::ASCEND_UB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 4);
        // L1
        l1_I = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(0);
        l1_X = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l1_Y = buf.template GetBuffer<BufferType::ASCEND_CB, InDtype>(chunk_size * chunk_size * sizeof(InDtype) * 2);

        // L0
        l0a_X = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(0);
        l0a_Y = buf.template GetBuffer<BufferType::ASCEND_L0A, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l0b_X = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(0);
        l0b_Y = buf.template GetBuffer<BufferType::ASCEND_L0B, InDtype>(chunk_size * chunk_size * sizeof(InDtype));
        l0c_X = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(0);
        l0c_Y = buf.template GetBuffer<BufferType::ASCEND_L0C, float>(chunk_size * chunk_size * sizeof(float));


        // Core
        num_core = AscendC::GetBlockNum();
        core_idx = AscendC::GetBlockIdx();
        sub_block_idx = AscendC::GetSubBlockIdx();
    }

    __aicore__ inline int64_t CeilDiv(int64_t a, int64_t b)
    {
        return (a + b - 1) / b;
    }

    __aicore__ inline void ub_to_l1(AscendC::LocalTensor<InDtype> l1Tensor,
                                    AscendC::LocalTensor<InDtype> ubTensor, uint32_t chunkSize)
    {
        AscendC::DataCopy(l1Tensor,                               // dst
                          ubTensor,                               // src
                          AscendC::DataCopyParams(1,              // nBurst
                                                  chunkSize * chunkSize / 16,  // lenBurst
                                                  0,              // srcGap
                                                  0));            // dstGap
    }

    __aicore__ inline void FixpipeL0cToL1(AscendC::LocalTensor<InDtype> l1Tensor,
                                          AscendC::LocalTensor<float> l0CTensor,
                                          uint32_t chunkSize)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixPipeParams;
        fixPipeParams.nSize = chunkSize;
        fixPipeParams.mSize = chunkSize;
        fixPipeParams.srcStride = chunkSize;
        fixPipeParams.dstStride = chunkSize * 16;
        
        // fixPipeParams.unitFlag = 0;
        if constexpr (std::is_same_v<InDtype, half>) {
            fixPipeParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixPipeParams.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<InDtype, float, CFG_NZ_L1>(l1Tensor, l0CTensor, fixPipeParams); 
    }

    __aicore__ inline void FixpipeL0cToUB(AscendC::LocalTensor<InDtype> ubTensor,
                                          AscendC::LocalTensor<float> l0CTensor,
                                          uint32_t chunkSize)
    {
        AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::NZ> fixPipeParams;
        fixPipeParams.nSize = chunkSize;
        fixPipeParams.mSize = chunkSize;
        fixPipeParams.srcStride = chunkSize;
        fixPipeParams.dstStride = chunkSize * 16;
        fixPipeParams.dualDstCtl = 0;
        fixPipeParams.subBlockId = 0;
        
        // fixPipeParams.unitFlag = 0;
        if constexpr (std::is_same_v<InDtype, half>) {
            fixPipeParams.quantPre = QuantMode_t::F322F16;
        } else {
            fixPipeParams.quantPre = QuantMode_t::F322BF16;
        }
        AscendC::Fixpipe<InDtype, float, CFG_NZ_UB>(ubTensor, l0CTensor, fixPipeParams); 
    }

    __aicore__ inline int64_t ChunkAlign(int64_t cur_chunk)
    {
        if (cur_chunk <= 16)
            return 16;
        if (cur_chunk <= 32)
            return 32;
        if (cur_chunk <= 64)
            return 64;
        return 128;
    }

    __aicore__ inline void MatmulToL0C(AscendC::LocalTensor<InDtype> l1A, AscendC::LocalTensor<InDtype> l1B,
                                       AscendC::LocalTensor<InDtype> l0A, AscendC::LocalTensor<InDtype> l0B,
                                       AscendC::LocalTensor<float> l0C, int64_t chunkSize, bool initC)
    {
        int64_t numFracs = chunkSize / 16;
        
        AscendC::LoadData2DParamsV2 loadDataParamsA;
        loadDataParamsA.mStartPosition = 0;
        loadDataParamsA.kStartPosition = 0;
        loadDataParamsA.mStep = numFracs; // 3
        loadDataParamsA.kStep = numFracs; // 3
        loadDataParamsA.srcStride = numFracs; // 3
        loadDataParamsA.dstStride = numFracs; // 3
        loadDataParamsA.ifTranspose = false;
        AscendC::LoadData(l0A, l1A, loadDataParamsA);

        AscendC::LoadData2DParamsV2 loadDataParamsB;
        loadDataParamsB.mStartPosition = 0; 
        loadDataParamsB.kStartPosition = 0; 
        loadDataParamsB.mStep = numFracs; 
        loadDataParamsB.kStep = numFracs; 
        loadDataParamsB.srcStride = numFracs; 
        loadDataParamsB.dstStride = numFracs;
        loadDataParamsB.ifTranspose = true;
        AscendC::LoadData(l0B, l1B, loadDataParamsB);

        PipeBarrier<PIPE_ALL>();

        AscendC::MmadParams mmadParams;
        mmadParams.m = chunkSize;
        mmadParams.n = chunkSize;
        mmadParams.k = chunkSize;
        mmadParams.cmatrixInitVal = initC;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = 0;
        AscendC::Mmad(l0C, l0A, l0B, mmadParams);
    }

    __aicore__ inline void AuxMatrixGen()
    {
        uint64_t NUM_FRACS = chunk_size / 16; // 对角矩阵个数
        uint64_t NUM_ITER = NUM_FRACS * 2;    // 切成8*16的块有多少
        int32_t chunkElems = chunk_size * chunk_size;
        // 清零A矩阵需要的内存
        Duplicate(ub_A, (InDtype)0, chunkElems);
        
        // 单位矩阵生成
        Duplicate(ub_I, (InDtype)0, chunkElems);
        for (uint64_t stripIdx = 0; stripIdx < NUM_ITER; stripIdx++) {
            uint64_t fracsIdx = stripIdx / 2;
            uint64_t oldEvenIdx = stripIdx % 2;
            uint64_t diagMask[2] = {
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][0],
                DIAG_MASK_8X16[oldEvenIdx ? 0 : 1][1]
            };
            uint64_t UB_DIAG_I_OFF = fracsIdx * (chunk_size + 16) * 16 + oldEvenIdx * 8 * 16;
            Duplicate(ub_I[UB_DIAG_I_OFF], (InDtype)1.0f, diagMask, 1, 1, 1);
        }
        // 全零矩阵生成
        Duplicate(ub_Zero, (InDtype)0, chunkElems);    
    }

    __aicore__ inline void ProcessVec()
    {
        for (uint64_t loop_idx = core_idx; loop_idx < chunk_num_total; loop_idx += num_core) {
            int64_t x_gm_offset = 0;
            int64_t seq_idx = 0;
            int64_t chunk_in_seq_idx = 0;
            int64_t head_idx = 0;
            int64_t chunk_idx = 0;

            // 2.计算索引
            if (mode == 1) { // 定长
                seq_idx = loop_idx / (chunk_num_in_seq * num_head);
                chunk_in_seq_idx = loop_idx % (chunk_num_in_seq * num_head) / num_head;
                head_idx = loop_idx % (chunk_num_in_seq * num_head) % num_head;
                x_gm_offset = seq_idx * num_head * seq_length * chunk_size +
                              chunk_in_seq_idx * num_head * chunk_size +
                              head_idx * chunk_size;
            } else if (mode == 2) { // 变长
                chunk_idx = loop_idx / num_head;
                seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2);
                seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
                chunk_num_in_seq = CeilDiv(seq_length, chunk_size);
                chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
                x_gm_offset = gm_cu_seqlens.GetValue(seq_idx) +
                              chunk_in_seq_idx * num_head * chunk_size +
                              head_idx * chunk_size;
            }

            chunk_size_actual = (chunk_in_seq_idx == (chunk_num_in_seq - 1)) ?
                                    ChunkAlign(seq_length - chunk_in_seq_idx * chunk_size) :
                                    chunk_size;

            // 3.MCH
            // gm_a -> ub_a
            AscendC::printf("wangwei[vector]: x_gm_offset %d\n", x_gm_offset);
            AscendC::printf("wangwei[vector]: chunk_size_actual %d\n", chunk_size_actual);

            // AscendC::DataCopy(ub_A, gm_a[x_gm_offset],
            //                   AscendC::Nd2NzParams(chunk_size_actual / 16,                  // ndNum
            //                                        16,                                      // nValue
            //                                        chunk_size_actual,                                      // dValue
            //                                        (chunk_size_actual * num_head + 1) * 16, // srcNdMatrixStride
            //                                        chunk_size_actual * num_head,            // srcDValue
            //                                        16,                                       // dstNzC0Stride
            //                                        1,                                       // dstNzNStride
            //                                        (chunk_size_actual + 16) * 16));          // dstNzMatrixStride
            
            for(uint64_t i = 0; i < chunk_size_actual / 16; i++){
                uint64_t srcOffset = i * (chunk_size_actual * 16 + 16);
                uint64_t dstOffset = i * (chunk_size_actual * 16 + 16 * 16);
                AscendC::DataCopy(ub_A[dstOffset], gm_a[x_gm_offset + srcOffset], AscendC::DataCopyParams(16, 1, 1, 0));     
            }
            
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::DataCopy(l1_Y,                               // dst
                          ub_A,                               // src
                          AscendC::DataCopyParams(1,              // nBurst
                                                  chunk_size_actual * chunk_size_actual / 16,  // lenBurst
                                                  0,              // srcGap
                                                  0));            // dstGap
            // AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1);
            // AscendC::DumpTensor(gm_a[x_gm_offset], 1, chunk_size_actual*chunk_size_actual); 
            // AscendC::DumpTensor(ub_A, 2, chunk_size_actual*chunk_size_actual); 
            
            AscendC::PipeBarrier<PIPE_ALL>();

            AscendC::Sub(ub_I_A, ub_I, ub_A, (int32_t)(chunk_size_actual * chunk_size_actual));

            

            AscendC::PipeBarrier<PIPE_ALL>();
            // SetFlag<HardEvent::V_MTE3>(0);
            // WaitFlag<HardEvent::V_MTE3>(0);
            // AscendC::DumpTensor(ub_I_A, 0, chunk_size_actual*chunk_size_actual);
            // AscendC::printf("wangwei: DataCopy(l1_X, ub_A)");

            AscendC::DataCopy(l1_X, ub_I_A,                                     // src
                          AscendC::DataCopyParams(1,                // nBurst
                                                  chunk_size_actual * chunk_size_actual / 16,  // lenBurst
                                                  0,                // srcGap
                                                  0));              // dstGap
            
            // AscendC::DumpTensor(ub_A, 1, chunk_size_actual*chunk_size_actual); 
            AscendC::PipeBarrier<PIPE_ALL>();

        }  
        
    }

    __aicore__ inline void ProcessCube()
    {
        for (uint64_t loop_idx = core_idx; loop_idx < chunk_num_total; loop_idx += num_core) {
            int64_t x_gm_offset = 0;
            int64_t seq_idx = 0;
            int64_t chunk_in_seq_idx = 0;
            int64_t head_idx = 0;
            int64_t chunk_idx = 0;

            // 2.计算索引
            if (mode == 1) { // 定长
                seq_idx = loop_idx / (chunk_num_in_seq * num_head);
                chunk_in_seq_idx = loop_idx % (chunk_num_in_seq * num_head) / num_head;
                head_idx = loop_idx % (chunk_num_in_seq * num_head) % num_head;
                x_gm_offset = seq_idx * num_head * seq_length * chunk_size +
                              chunk_in_seq_idx * num_head * chunk_size +
                              head_idx * chunk_size;
            } else if (mode == 2) { // 变长
                chunk_idx = loop_idx / num_head;
                seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2);
                seq_length = gm_cu_seqlens.GetValue(seq_idx + 1) - gm_cu_seqlens.GetValue(seq_idx);
                chunk_num_in_seq = CeilDiv(seq_length, chunk_size);
                chunk_in_seq_idx = gm_chunk_indices.GetValue(chunk_idx * 2 + 1);
                x_gm_offset = gm_cu_seqlens.GetValue(seq_idx) +
                              chunk_in_seq_idx * num_head * chunk_size +
                              head_idx * chunk_size;
            }
            chunk_size_actual = (chunk_in_seq_idx == (chunk_num_in_seq - 1)) ?
                                    ChunkAlign(seq_length - chunk_in_seq_idx * chunk_size) :
                                    chunk_size;
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE1>(0x2);
            
            // AscendC::printf("wangwei: l1_X: \n");
            // AscendC::DumpTensor(l1_X, 1, chunk_size_actual * chunk_size_actual);

            MatmulToL0C(l1_Y, l1_Y, l0a_Y, l0b_Y, l0c_Y, chunk_size_actual, true);
            AscendC::PipeBarrier<PIPE_ALL>();
            FixpipeL0cToL1(l1_Y, l0c_Y, chunk_size_actual);
            AscendC::PipeBarrier<PIPE_ALL>();

            MatmulToL0C(l1_I, l1_X, l0a_X, l0b_X, l0c_X, chunk_size_actual, true);
            AscendC::PipeBarrier<PIPE_ALL>();
            FixpipeL0cToL1(l1_X, l0c_X, chunk_size_actual);
            AscendC::PipeBarrier<PIPE_ALL>();

            MatmulToL0C(l1_X, l1_Y, l0a_X, l0b_X, l0c_X, chunk_size_actual, false);
            AscendC::PipeBarrier<PIPE_ALL>();
            FixpipeL0cToL1(l1_X, l0c_X, chunk_size_actual);
            AscendC::PipeBarrier<PIPE_ALL>();

            for (uint64_t iter = 0; iter < 2; iter++) {
                MatmulToL0C(l1_Y, l1_Y, l0a_Y, l0b_Y, l0c_Y, chunk_size_actual, true);
                AscendC::PipeBarrier<PIPE_ALL>();
                FixpipeL0cToL1(l1_Y, l0c_Y, chunk_size_actual);
                AscendC::PipeBarrier<PIPE_ALL>();
                MatmulToL0C(l1_X, l1_Y, l0a_X, l0b_X, l0c_X, chunk_size_actual, false);
                AscendC::PipeBarrier<PIPE_ALL>();
                if(iter == 1){
                    FixpipeL0cToUB(ub_Res, l0c_X, chunk_size_actual);
                    AscendC::PipeBarrier<PIPE_ALL>();
                    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x0);
                } else {
                    FixpipeL0cToL1(l1_X, l0c_X, chunk_size_actual);
                    AscendC::PipeBarrier<PIPE_ALL>();
                }
                // FixpipeL0cToL1(l1_X, l0c_X, chunk_size_actual);
                
            }
            

            
            // AscendC::SyncAll<false>();
            // AscendC::DumpTensor(l1_X, 9, chunk_size_actual * chunk_size_actual);
        }  
        
    }

    __aicore__ inline void Process(){
        
        if ASCEND_IS_AIV {
            if (sub_block_idx == 0){
                // AscendC::DumpTensor(gm_a, 9, chunk_size * chunk_size);
                // AscendC::printf("wangwei: ASCEND_IS_AIV");
                AuxMatrixGen();
                AscendC::PipeBarrier<PIPE_ALL>();
                

                AscendC::DataCopy(l1_I, ub_I,                           // src
                              AscendC::DataCopyParams(1,                // nBurst
                                                      chunk_size * chunk_size / 16,  // lenBurst
                                                      0,                // srcGap
                                                      0));              // dstGap
                // AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x0);
                            
                ProcessVec();
            }
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x2);
            AscendC::CrossCoreWaitFlag<0x2>(0x0);
            // AscendC::CrossCoreWaitFlag(0x0);
            // AscendC::PipeBarrier<PIPE_ALL>();
            // AscendC::SyncAll<false>();
            // AscendC::Muls(ub_Res, ub_Res, 1, (int32_t)(chunk_size * chunk_size));
            AscendC::DumpTensor(ub_Res, 3, chunk_size * chunk_size);
            
            
        }
        if ASCEND_IS_AIC {
            // AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(0x0);
            // AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(0x1);
            
            // AscendC::DumpTensor(l1_X, 0, chunk_size * chunk_size); 
            // AscendC::DumpTensor(l1_Y, 1, chunk_size * chunk_size); 
            // AscendC::DumpTensor(l1_I, 2, chunk_size * chunk_size); 
            ProcessCube();
        }
    }
private:
    // Gm
    AscendC::GlobalTensor<InDtype> gm_a;
    AscendC::GlobalTensor<int32_t> gm_cu_seqlens;
    AscendC::GlobalTensor<int32_t> gm_chunk_indices;
    AscendC::GlobalTensor<InDtype> gm_out;

    // UB
    AscendC::LocalTensor<InDtype> ub_A;
    AscendC::LocalTensor<InDtype> ub_I_A;
    AscendC::LocalTensor<InDtype> ub_I;
    AscendC::LocalTensor<InDtype> ub_Zero;
    AscendC::LocalTensor<InDtype> ub_Res;

    // L1
    AscendC::LocalTensor<InDtype> l1_X;
    AscendC::LocalTensor<InDtype> l1_Y;
    AscendC::LocalTensor<InDtype> l1_I;

    // L0
    AscendC::LocalTensor<InDtype> l0a_X;
    AscendC::LocalTensor<InDtype> l0a_Y;
    AscendC::LocalTensor<InDtype> l0b_X;
    AscendC::LocalTensor<InDtype> l0b_Y;
    AscendC::LocalTensor<float> l0c_X;
    AscendC::LocalTensor<float> l0c_Y;

    // Tiling
    int64_t batch_size;
    int64_t seq_length;
    int64_t num_head;
    int64_t chunk_size;
    int64_t chunk_size_actual;
    int64_t chunk_num_in_seq;
    int64_t chunk_num_total;
    int64_t mode;

    // Core
    int64_t num_core;
    int64_t core_idx;
    int64_t sub_block_idx;
};


#endif  // SOLVE_TRIL_H
