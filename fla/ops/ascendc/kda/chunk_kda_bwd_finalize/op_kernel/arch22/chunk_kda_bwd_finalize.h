/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */
#ifndef CHUNK_KDA_BWD_FINALIZE_ARCH22_H
#define CHUNK_KDA_BWD_FINALIZE_ARCH22_H
#include "chunk_kda_bwd_finalize_cube.h"
#include "chunk_kda_bwd_finalize_vector.h"
#include "chunk_kda_bwd_finalize_gate.h"

namespace KDA::Arch22 {
// Two workspace generations overlap adjacent work items. Stage 12 reduces
// the parameter gradients after the AIV-only barrier in this same launch.
constexpr uint16_t VECTOR_RELEASE = 9;
constexpr uint16_t VECTOR_ACK = 10;
constexpr uint16_t S02_READY = 4;
constexpr uint16_t S5_DONE = 12; // Stage 7 builds release after the final band.
constexpr uint16_t S6_READY = 6;

__aicore__ inline ChunkInfo ResolveChunk(const FinalizeTilingData &tiling, const __gm__ int64_t *cu,
                                         const __gm__ int64_t *indices,
                                         int64_t chunkIndex)
{
    ChunkInfo chunk;
    if (tiling.isVariable) {
        chunk.sequence = indices[2 * chunkIndex];
        chunk.localChunk = indices[2 * chunkIndex + 1];
        chunk.tokenStart = cu[chunk.sequence] + chunk.localChunk * CHUNK_SIZE;
        const int64_t remaining = cu[chunk.sequence + 1] - chunk.tokenStart;
        chunk.validRows = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
    } else {
        const int64_t chunksPerSequence = (tiling.tokens + CHUNK_SIZE - 1) / CHUNK_SIZE;
        chunk.sequence = chunkIndex / chunksPerSequence;
        chunk.localChunk = chunkIndex % chunksPerSequence;
        chunk.tokenStart = chunk.localChunk * CHUNK_SIZE;
        const int64_t remaining = tiling.tokens - chunk.tokenStart;
        chunk.validRows = remaining < CHUNK_SIZE ? remaining : CHUNK_SIZE;
    }
    return chunk;
}

template <bool Q_NORM, typename BetaT>
__aicore__ inline void RunFinalize(GM_ADDR k, GM_ADDR v, GM_ADDR gk, GM_ADDR akk, GM_ADDR vNew, GM_ADDR h,
                                         GM_ADDR dh, GM_ADDR dvScan, GM_ADDR beta, GM_ADDR dv, GM_ADDR q, GM_ADDR dqRaw,
                                         float scale, GM_ADDR dAqk, GM_ADDR dq, GM_ADDR qRstd, GM_ADDR cuSeqlens, GM_ADDR chunkIndices,
                                         GM_ADDR workspace, GM_ADDR capture, GM_ADDR trace, FinalizeTilingData tiling,
                                         FinalizeTailArgs tail)
{
    constexpr uint32_t LAST_STAGE = 12;
    constexpr bool CAPTURE = false;
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    const int64_t workCount = ((tiling.heads + 1) / HEADS_PER_WORK) * tiling.chunks;
    const auto *cu = reinterpret_cast<const __gm__ int64_t *>(cuSeqlens);
    const auto *indices = reinterpret_cast<const __gm__ int64_t *>(chunkIndices);
    AscendC::TPipe pipe;
    if ASCEND_IS_AIC {
        ChunkKdaBwdFinalizeCube cube;
        cube.Init(v, akk, vNew, h, dh, dvScan, workspace, &pipe);
        for (int64_t work = core; work < workCount; work += tiling.usedCoreNum) {
            const uint32_t bank = LAST_STAGE >= 3 ? (work / tiling.usedCoreNum) & 1 : 0;
            const int64_t chunkIndex = work % tiling.chunks;
            const int64_t firstHead = (work / tiling.chunks) * HEADS_PER_WORK;
            // Fixed-length metadata stays in registers; variable lengths use the
            // packed table.
            const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, chunkIndex);
            if (LAST_STAGE < 2 || work == core) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(WORK_OPEN);
            }
            if constexpr (LAST_STAGE >= 5) {
                if (work >= core + 2 * tiling.usedCoreNum) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S5_DONE + bank);
                }
            }
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner < tiling.heads) {
                    cube.LoadStage0Inputs(tiling, chunk, chunkIndex, firstHead + owner, owner, work == core,
                                          LAST_STAGE >= 4 ? bank : 0);
                }
            }
            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                if (firstHead + owner < tiling.heads) {
                    cube.Stage0<CAPTURE, (LAST_STAGE >= 5 && !CAPTURE)>(tiling, chunk, chunkIndex, firstHead + owner,
                                                                        owner, core, capture, bank,
                                                                        LAST_STAGE >= 4 ? bank : 0);
                    if constexpr (LAST_STAGE < 2) {
                        cube.WaitDw(owner);
                        cube.ReleaseAkk(owner);
                    }
                    if constexpr (LAST_STAGE >= 2) {
                        if (owner + 1 == HEADS_PER_WORK || firstHead + owner + 1 == tiling.heads) {
                            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(KE_READY);
                            for (uint32_t keOwner = 0; keOwner < HEADS_PER_WORK; ++keOwner) {
                                if (firstHead + keOwner < tiling.heads) {
                                    cube.LoadKe(keOwner, core, bank);
                                }
                            }
                            // kE is now protected in L1. Vector can prepare the next work
                            // while Cube consumes the current one; akk has its own free
                            // event.
                            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(VECTOR_RELEASE);
                            if (work + tiling.usedCoreNum < workCount) {
                                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(WORK_OPEN);
                            }
                        }
                    }
                    // These five inputs end their L1 lifetime in Stage 0.
                    // Keep akk and dW untouched for their later consumers.
                    const int64_t nextWork = work + tiling.usedCoreNum;
                    if (nextWork < workCount) {
                        const int64_t nextChunkIndex = nextWork % tiling.chunks;
                        const int64_t nextHead = (nextWork / tiling.chunks) * HEADS_PER_WORK + owner;
                        if (nextHead < tiling.heads) {
                            const ChunkInfo nextChunk = ResolveChunk(tiling, cu, indices, nextChunkIndex);
                            cube.PreloadState(tiling, nextChunk, nextChunkIndex, nextHead, owner);
                        }
                    }
                }
            }
            if constexpr (LAST_STAGE < 2) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(KE_READY);
            }
            if constexpr (LAST_STAGE >= 2) {
                for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                    if (firstHead + owner < tiling.heads) {
                        cube.Stage2<CAPTURE>(tiling, chunkIndex, firstHead + owner, owner, core, capture, bank,
                                             LAST_STAGE >= 4);
                    }
                }
            }
            if constexpr (LAST_STAGE < 2) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE1>(VECTOR_RELEASE);
            }
            if constexpr (LAST_STAGE >= 3) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(S02_READY + bank);
                // Two GM generations: retire the previous consumer before reusing its
                // bank.
                if (work != core) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_ACK + (bank ^ 1));
                    if constexpr (LAST_STAGE >= 4) {
                        const int64_t readyWork = work - tiling.usedCoreNum;
                        const int64_t readyHead = (readyWork / tiling.chunks) * HEADS_PER_WORK;
                        const uint32_t readyBank = (readyWork / tiling.usedCoreNum) & 1;
                        for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                            if (readyHead + owner < tiling.heads) {
                                cube.LoadZb(owner, core, readyBank);
                            }
                        }
                        for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                            if (readyHead + owner < tiling.heads) {
                                cube.Stage4<CAPTURE, (LAST_STAGE >= 6)>(tiling, readyWork % tiling.chunks,
                                                                        readyHead + owner, owner, readyBank, capture);
                            }
                        }
                        if constexpr (LAST_STAGE >= 6) {
                            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                                if (readyHead + owner < tiling.heads) {
                                    cube.Stage6<CAPTURE>(tiling, readyWork % tiling.chunks, readyHead + owner, owner,
                                                         core, readyBank, capture);
                                }
                            }
                            if constexpr (LAST_STAGE >= 7) {
                                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(S6_READY + readyBank);
                                if constexpr (LAST_STAGE >= 8) {
                                    const int64_t readyChunk = readyWork % tiling.chunks;
                                    cube.Stage8<CAPTURE, (LAST_STAGE >= 9), (LAST_STAGE >= 10), (LAST_STAGE >= 11)>(
                                        tiling, ResolveChunk(tiling, cu, indices, readyChunk), readyChunk, readyHead, core,
                                        readyBank, capture);
                                }
                            }
                        }
                    }
                }
                if (work + tiling.usedCoreNum >= workCount) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_ACK + bank);
                    if constexpr (LAST_STAGE >= 4) {
                        const int64_t readyWork = work;
                        const int64_t readyHead = (readyWork / tiling.chunks) * HEADS_PER_WORK;
                        const uint32_t readyBank = (readyWork / tiling.usedCoreNum) & 1;
                        for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                            if (readyHead + owner < tiling.heads) {
                                cube.LoadZb(owner, core, readyBank);
                            }
                        }
                        for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                            if (readyHead + owner < tiling.heads) {
                                cube.Stage4<CAPTURE, (LAST_STAGE >= 6)>(tiling, readyWork % tiling.chunks,
                                                                        readyHead + owner, owner, readyBank, capture);
                            }
                        }
                        if constexpr (LAST_STAGE >= 6) {
                            for (uint32_t owner = 0; owner < HEADS_PER_WORK; ++owner) {
                                if (readyHead + owner < tiling.heads) {
                                    cube.Stage6<CAPTURE>(tiling, readyWork % tiling.chunks, readyHead + owner, owner,
                                                         core, readyBank, capture);
                                }
                            }
                            if constexpr (LAST_STAGE >= 7) {
                                AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(S6_READY + readyBank);
                                if constexpr (LAST_STAGE >= 8) {
                                    const int64_t readyChunk = readyWork % tiling.chunks;
                                    cube.Stage8<CAPTURE, (LAST_STAGE >= 9), (LAST_STAGE >= 10), (LAST_STAGE >= 11)>(
                                        tiling, ResolveChunk(tiling, cu, indices, readyChunk), readyChunk, readyHead, core,
                                        readyBank, capture);
                                }
                            }
                        }
                    }
                }
            } else {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_ACK);
            }
        }
        if constexpr (LAST_STAGE >= 5) {
            const int64_t last = core + ((workCount - 1 - core) / tiling.usedCoreNum) * tiling.usedCoreNum;
            if (last > core) {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S5_DONE + (((last / tiling.usedCoreNum) & 1) ^ 1));
            }
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S5_DONE + ((last / tiling.usedCoreNum) & 1));
        }
        cube.End();
    }
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetSubBlockIdx();
        constexpr bool COMPACT_MASK = LAST_STAGE >= 11 && !CAPTURE && !Q_NORM;
        AscendC::TBuf<AscendC::TPosition::VECCALC> ubBuf;
        pipe.InitBuffer(ubBuf, UB_BYTES);
        auto ub = ubBuf.Get<uint8_t>();
        auto record = ub[85 * 1024].ReinterpretCast<int64_t>();
        ChunkKdaBwdFinalizeVector<BetaT> vector;
        if constexpr (LAST_STAGE >= 1) {
            vector.Init(k, gk, h, dh, workspace, ub, &pipe, tiling);
            if constexpr (LAST_STAGE >= 3) {
                vector.template InitStage3<COMPACT_MASK>(v, beta, dv);
            }
            if constexpr (LAST_STAGE >= 5) {
                vector.InitStage5(q, dqRaw, scale);
                if constexpr (LAST_STAGE >= 7) {
                    vector.InitStage7(dAqk);
                    if constexpr (LAST_STAGE >= 9) {
                        vector.InitStage9(dq, qRstd);
                        if constexpr (LAST_STAGE >= 11) {
                            vector.InitStage11(tail);
                        }
                    }
                }
            }
        }
        AscendC::GlobalTensor<int64_t> traceGm;
        traceGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(trace));
        const auto traceReady = pipe.AllocEventID<AscendC::HardEvent::S_MTE3>();
        const auto traceFree = pipe.AllocEventID<AscendC::HardEvent::MTE3_S>();
        if constexpr (CAPTURE && LAST_STAGE >= 3) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(traceFree);
        }
        for (int64_t work = core; work < workCount; work += tiling.usedCoreNum) {
            const uint32_t bank = LAST_STAGE >= 3 ? (work / tiling.usedCoreNum) & 1 : 0;
            const int64_t chunkIndex = work % tiling.chunks;
            const int64_t head = (work / tiling.chunks) * HEADS_PER_WORK + owner;
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(WORK_OPEN);
            if constexpr (LAST_STAGE >= 1) {
                if (head < tiling.heads) {
                    const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, chunkIndex);
                    if (chunk.validRows == CHUNK_SIZE) {
                        vector.template Stage1<CAPTURE, (LAST_STAGE >= 5), true, (LAST_STAGE >= 5 && !CAPTURE)>(
                            tiling, chunk, chunkIndex, head, owner, core, capture, bank);
                    } else {
                        vector.template Stage1<CAPTURE, (LAST_STAGE >= 5), false, (LAST_STAGE >= 5 && !CAPTURE)>(
                            tiling, chunk, chunkIndex, head, owner, core, capture, bank);
                    }
                }
            }
            // The absent owner in an odd-head work also closes the collective.
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(KE_READY);
            if constexpr (LAST_STAGE >= 3) {
                // Send the next kE first; retire the preceding Stage 3 while Cube runs
                // Stage 2.
                if (work != core) {
                    const int64_t previous = work - tiling.usedCoreNum;
                    const int64_t previousChunk = previous % tiling.chunks;
                    const int64_t previousHead = (previous / tiling.chunks) * HEADS_PER_WORK + owner;
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S02_READY + (bank ^ 1));
                    if (previousHead < tiling.heads) {
                        const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, previousChunk);
                        if (chunk.validRows == CHUNK_SIZE) {
                            vector.template Stage3<CAPTURE, true, COMPACT_MASK>(
                                tiling, chunk, previousChunk, previousHead, owner, core, capture, bank ^ 1);
                        } else {
                            vector.template Stage3<CAPTURE, false, COMPACT_MASK>(
                                tiling, chunk, previousChunk, previousHead, owner, core, capture, bank ^ 1);
                        }
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(VECTOR_ACK + (bank ^ 1));
                    if constexpr (LAST_STAGE >= 5) {
                        if (previousHead < tiling.heads) {
                            const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, previousChunk);
                            if (chunk.validRows == CHUNK_SIZE) {
                                vector.template Stage5<CAPTURE, true, !CAPTURE, (LAST_STAGE >= 11 && !CAPTURE && !Q_NORM),
                                              (LAST_STAGE >= 11 && !CAPTURE)>(
                                    tiling, chunk, previousChunk, previousHead, owner, core, capture, bank ^ 1);
                            } else {
                                vector.template Stage5<CAPTURE, false, !CAPTURE>(tiling, chunk, previousChunk, previousHead,
                                                                        owner, core, capture, bank ^ 1);
                            }
                        }
                        if constexpr (LAST_STAGE >= 7) {
                            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S6_READY + (bank ^ 1));
                            if (previousHead < tiling.heads) {
                                const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, previousChunk);
                                vector.template Stage7<CAPTURE, (LAST_STAGE >= 8), (LAST_STAGE >= 9), Q_NORM, (LAST_STAGE >= 10),
                                              (LAST_STAGE >= 11)>(tiling, chunk, previousChunk, previousHead, owner,
                                                                  core, capture, bank ^ 1);
                            } else if constexpr (LAST_STAGE >= 8) {
                                const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, previousChunk);
                                for (int32_t start = ((chunk.validRows - 1) / 32) * 32; start >= 0; start -= 32) {
                                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(BAND_READY);
                                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(BAND_LOADED);
                                    if constexpr (LAST_STAGE >= 10) {
                                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(BAND_READY);
                                    }
                                    if constexpr (LAST_STAGE >= 9) {
                                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S8_READY);
                                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S8_SECOND_READY);
                                    }
                                    if constexpr (LAST_STAGE >= 10) {
                                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(S10_FIRST_READY);
                                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(BAND_LOADED);
                                    }
                                }
                            }
                        }
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(S5_DONE + (bank ^ 1));
                    }
                }
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_RELEASE);
                if (work + tiling.usedCoreNum >= workCount) {
                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S02_READY + bank);
                    if (head < tiling.heads) {
                        const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, chunkIndex);
                        if (chunk.validRows == CHUNK_SIZE) {
                            vector.template Stage3<CAPTURE, true, COMPACT_MASK>(
                                tiling, chunk, chunkIndex, head, owner, core, capture, bank);
                        } else {
                            vector.template Stage3<CAPTURE, false, COMPACT_MASK>(
                                tiling, chunk, chunkIndex, head, owner, core, capture, bank);
                        }
                    }
                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(VECTOR_ACK + bank);
                    if constexpr (LAST_STAGE >= 5) {
                        if (head < tiling.heads) {
                            const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, chunkIndex);
                            if (chunk.validRows == CHUNK_SIZE) {
                                vector.template Stage5<CAPTURE, true, !CAPTURE, (LAST_STAGE >= 11 && !CAPTURE && !Q_NORM),
                                              (LAST_STAGE >= 11 && !CAPTURE)>(
                                    tiling, chunk, chunkIndex, head, owner, core, capture, bank);
                            } else {
                                vector.template Stage5<CAPTURE, false, !CAPTURE>(tiling, chunk, chunkIndex, head, owner, core,
                                                                        capture, bank);
                            }
                        }
                        if constexpr (LAST_STAGE >= 7) {
                            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S6_READY + bank);
                            if (head < tiling.heads) {
                                const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, chunkIndex);
                                vector.template Stage7<CAPTURE, (LAST_STAGE >= 8), (LAST_STAGE >= 9), Q_NORM, (LAST_STAGE >= 10),
                                              (LAST_STAGE >= 11)>(tiling, chunk, chunkIndex, head, owner, core, capture,
                                                                  bank);
                            } else if constexpr (LAST_STAGE >= 8) {
                                const ChunkInfo chunk = ResolveChunk(tiling, cu, indices, chunkIndex);
                                for (int32_t start = ((chunk.validRows - 1) / 32) * 32; start >= 0; start -= 32) {
                                    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(BAND_READY);
                                    AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(BAND_LOADED);
                                    if constexpr (LAST_STAGE >= 10) {
                                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(BAND_READY);
                                    }
                                    if constexpr (LAST_STAGE >= 9) {
                                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S8_READY);
                                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(S8_SECOND_READY);
                                    }
                                    if constexpr (LAST_STAGE >= 10) {
                                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(S10_FIRST_READY);
                                        AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>(BAND_LOADED);
                                    }
                                }
                            }
                        }
                        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(S5_DONE + (bank));
                    }
                }
            } else {
                AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE2>(VECTOR_RELEASE);
            }
            if constexpr (CAPTURE) {
                if constexpr (LAST_STAGE >= 3) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(traceFree);
                }
                record.SetValue(0, work);
                record.SetValue(1, core);
                record.SetValue(2, owner);
                record.SetValue(3, head);
                record.SetValue(4, chunkIndex);
                record.SetValue(5, work / tiling.usedCoreNum + 1);
                record.SetValue(6, tiling.usedCoreNum);
                record.SetValue(7, ResolveChunk(tiling, cu, indices, chunkIndex).validRows);
                record.SetValue(8, head < tiling.heads ? 1 : 0);
                record.SetValue(9, LAST_STAGE);
                record.SetValue(10, 7); // WORK_OPEN waited, KE_READY sent, VECTOR_RELEASE waited.
                record.SetValue(11, AscendC::GetBlockNum());
                record.SetValue(12, AscendC::GetSubBlockNum());
                record.SetValue(13, AscendC::GetBlockIdx());
                record.SetValue(14, static_cast<int64_t>(core) * WS_CORE_BYTES + owner * WS_HEAD_BYTES +
                                        bank * WS_BANK_BYTES);
                record.SetValue(15, 0);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(traceReady);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(traceReady);
                AscendC::DataCopy(traceGm[(work * HEADS_PER_WORK + owner) * 16], record, 16);
                if constexpr (LAST_STAGE >= 3) {
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(traceFree);
                }
            }
            if constexpr (LAST_STAGE < 3) {
                AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(VECTOR_ACK);
            }
            if constexpr (LAST_STAGE < 3) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(traceFree);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(traceFree);
            }
        }
        if constexpr (CAPTURE && LAST_STAGE >= 3) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(traceFree);
        }
        pipe.ReleaseEventID<AscendC::HardEvent::S_MTE3>(traceReady);
        pipe.ReleaseEventID<AscendC::HardEvent::MTE3_S>(traceFree);
        if constexpr (LAST_STAGE >= 1) {
            vector.End();
        }
        if constexpr (LAST_STAGE == 12) {
            // Every Stage 8 flag-14 notification has been consumed, including
            // the absent owner of an odd-head work. All partial writes finish
            // before the AIV-only barrier reuses flag 14.
            AscendC::SyncAll<true>();
            FinalizeStage12(tail, tiling, pipe, ub.ReinterpretCast<float>(),
                            AscendC::GetBlockIdx(), tiling.usedCoreNum * HEADS_PER_WORK);
        }
    }
}


} // namespace KDA::Arch22
#endif
