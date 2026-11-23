// Boxed Torch Stable ABI adapter for the optional 910B inline kernels.
#include "stable/boxed.h"
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include <algorithm>
#include <climits>
#include <initializer_list>
#include <stdexcept>
#include <string>

extern "C" int aclrtGetDevice(int32_t *);
extern "C" int launch_scan(float *, float *, float *, int, int, int, void *);
extern "C" int launch_bridge(float *, float *, int64_t *, float *, void *, int, int, int, int, int64_t, int64_t, int,
                             int, void *);
extern "C" int launch_diagonal(float *, float *, int, int, void *);

namespace {
using fla_npu_stable::stable::boxed_adapter;
using torch::headeronly::ScalarType;
using torch::stable::Tensor;

void require(bool condition, const char *message)
{
    if (!condition)
        throw std::invalid_argument(message);
}

void check_tensor(const Tensor &tensor, int device, ScalarType dtype, std::initializer_list<int64_t> shape,
                  bool contiguous = true)
{
    int32_t type;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(tensor.get(), &type));
    require(type == aoti_torch_device_type_privateuse1(), "VDN Native: expected an NPU tensor");
    require(tensor.get_device_index() == device && tensor.scalar_type() == dtype, "VDN Native: wrong device or dtype");
    require(tensor.dim() == static_cast<int64_t>(shape.size()), "VDN Native: wrong rank");
    require(!contiguous || tensor.is_contiguous(), "VDN Native: expected contiguous tensor");
    int index = 0;
    for (int64_t size : shape) {
        require(tensor.size(index++) == size, "VDN Native: incompatible tensor shape");
    }
}

void check_device(int expected)
{
    int32_t current = -1;
    require(aclrtGetDevice(&current) == 0 && current == expected, "VDN Native: expected input device to be current");
}

void check_launch(int status)
{
    if (status != 0)
        throw std::runtime_error("VDN kernel launch failed: " + std::to_string(status));
}

std::tuple<Tensor, Tensor> scan(Tensor transition, Tensor prefix, Tensor suffix, int64_t stream)
{
    require(transition.dim() == 4, "VDN scan expects [frames, heads, 128, 128]");
    const auto frames = transition.size(0), heads = transition.size(1);
    require(frames > 0 && frames < INT_MAX && heads > 0 && heads < INT_MAX / 2, "VDN scan shape out of range");
    const auto device = transition.get_device_index();
    check_tensor(transition, device, ScalarType::Float, {frames, heads, 128, 128});
    check_tensor(prefix, device, ScalarType::Float, {frames + 1, heads, 128, 128});
    check_tensor(suffix, device, ScalarType::Float, {frames + 1, heads, 128, 128});
    check_device(device);
    check_launch(launch_scan(static_cast<float *>(transition.data_ptr()), static_cast<float *>(prefix.data_ptr()),
                             static_cast<float *>(suffix.data_ptr()), frames, heads, std::min<int64_t>(24, 2 * heads),
                             reinterpret_cast<void *>(stream)));
    return {prefix, suffix};
}

Tensor bridge(Tensor prefix, Tensor suffix, Tensor bounds, Tensor logs, Tensor out, int64_t start, int64_t stream)
{
    require(prefix.dim() == 4 && out.dim() == 4, "VDN bridge expects rank-four banks and output");
    const auto frames = prefix.size(0) - 1, heads = prefix.size(1), count = out.size(0);
    require(frames > 0 && frames < INT_MAX && count > 0 && count <= frames && start >= 0 && start <= frames - count &&
                heads > 0 && heads <= INT_MAX / count,
            "VDN bridge shape out of range");
    const auto device = prefix.get_device_index();
    check_tensor(prefix, device, ScalarType::Float, {frames + 1, heads, 128, 128});
    check_tensor(suffix, device, ScalarType::Float, {frames + 1, heads, 128, 128});
    check_tensor(bounds, device, ScalarType::Long, {frames, 2}, false);
    check_tensor(logs, device, ScalarType::Float, {frames + 1, heads, 128});
    const auto dtype = out.scalar_type();
    require(dtype == ScalarType::Float || dtype == ScalarType::BFloat16, "VDN bridge expects FP32 or BF16 output");
    check_tensor(out, device, dtype, {count, heads, 128, 128});
    check_device(device);
    check_launch(launch_bridge(static_cast<float *>(prefix.data_ptr()), static_cast<float *>(suffix.data_ptr()),
                               static_cast<int64_t *>(bounds.data_ptr()), static_cast<float *>(logs.data_ptr()),
                               out.data_ptr(), start, count, heads, frames, bounds.stride(0), bounds.stride(1),
                               dtype == ScalarType::BFloat16, std::min<int64_t>(48, count * heads),
                               reinterpret_cast<void *>(stream)));
    return out;
}

Tensor add_diagonal(Tensor correction, Tensor two_eye, int64_t stream)
{
    require(correction.dim() == 3, "VDN correction expects rank three");
    const auto count = correction.size(0);
    require(count > 0 && count <= INT_MAX / 2, "VDN correction batch out of range");
    const auto device = correction.get_device_index();
    check_tensor(correction, device, ScalarType::Float, {count, 128, 128});
    check_tensor(two_eye, device, ScalarType::Float, {128, 128});
    check_device(device);
    check_launch(launch_diagonal(static_cast<float *>(correction.data_ptr()), static_cast<float *>(two_eye.data_ptr()),
                                 count, std::min<int64_t>(48, 2 * count), reinterpret_cast<void *>(stream)));
    return correction;
}
} // namespace

STABLE_TORCH_LIBRARY(fla_vdn_910b, m)
{
    m.def("scan(Tensor transition, Tensor(a!) prefix, Tensor(b!) suffix, int stream) -> (Tensor(a!), Tensor(b!))");
    m.def("bridge(Tensor prefix, Tensor suffix, Tensor bounds, Tensor logs, Tensor(a!) out, int start, int stream) -> "
          "Tensor(a!)");
    m.def("add_diagonal(Tensor(a!) correction, Tensor two_eye, int stream) -> Tensor(a!)");
}

STABLE_TORCH_LIBRARY_IMPL(fla_vdn_910b, PrivateUse1, m)
{
    m.impl("scan", &boxed_adapter<scan>);
    m.impl("bridge", &boxed_adapter<bridge>);
    m.impl("add_diagonal", &boxed_adapter<add_diagonal>);
}

