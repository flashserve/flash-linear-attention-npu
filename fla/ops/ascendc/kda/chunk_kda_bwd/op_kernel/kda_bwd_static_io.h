#ifndef KDA_BWD_STATIC_IO_H
#define KDA_BWD_STATIC_IO_H
#include "kernel_operator.h"

namespace KDA {
template <uint32_t Bytes, bool Input>
class KdaBwdStaticIo {
    static constexpr AscendC::HardEvent kReady = Input ?
        AscendC::HardEvent::MTE2_V : AscendC::HardEvent::V_MTE3;
    static constexpr AscendC::HardEvent kFree = Input ?
        AscendC::HardEvent::V_MTE2 : AscendC::HardEvent::MTE3_V;
public:
    __aicore__ inline void Init(AscendC::TPipe *pipe)
    {
        pipe_ = pipe;
        pipe_->InitBuffer(storage_, 2 * Bytes);
        for (uint32_t i = 0; i < 2; ++i) {
            ready_[i] = static_cast<event_t>(pipe_->AllocEventID<kReady>());
            free_[i] = static_cast<event_t>(pipe_->AllocEventID<kFree>());
            AscendC::SetFlag<kFree>(free_[i]);
        }
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> AllocTensor()
    {
        AscendC::WaitFlag<kFree>(free_[slot_]);
        return storage_.template Get<T>()[slot_ * (Bytes / sizeof(T))];
    }

    template <typename T>
    __aicore__ inline void EnQue(AscendC::LocalTensor<T>)
    {
        AscendC::SetFlag<kReady>(ready_[slot_]);
    }

    template <typename T>
    __aicore__ inline AscendC::LocalTensor<T> DeQue()
    {
        AscendC::WaitFlag<kReady>(ready_[slot_]);
        return storage_.template Get<T>()[slot_ * (Bytes / sizeof(T))];
    }

    template <typename T>
    __aicore__ inline void FreeTensor(AscendC::LocalTensor<T>)
    {
        AscendC::SetFlag<kFree>(free_[slot_]);
        slot_ ^= 1U;
    }

    __aicore__ inline ~KdaBwdStaticIo()
    {
        if (pipe_ != nullptr) {
            for (uint32_t i = 0; i < 2; ++i) {
                AscendC::WaitFlag<kFree>(free_[i]);
                pipe_->ReleaseEventID<kReady>(ready_[i]);
                pipe_->ReleaseEventID<kFree>(free_[i]);
            }
        }
    }
private:
    AscendC::TPipe *pipe_ = nullptr;
    AscendC::TBuf<AscendC::TPosition::VECCALC> storage_;
    event_t ready_[2]{};
    event_t free_[2]{};
    uint32_t slot_ = 0;
};
}
#endif
