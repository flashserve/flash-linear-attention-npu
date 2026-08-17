#pragma once

#include "kernel_operator.h"

// Keep compatibility fixes local to KDA while shared CATLASS helpers remain unchanged.
namespace Common {
using AscendC::SizeOfBits;
}

namespace Catlass::Gemm::Block {
using AscendC::SizeOfBits;
}
