// Minimal PyTorch CUDA extension — a single elementwise-add kernel and its
// pybind11 binding. Sanity-checks the Windows build toolchain (MSVC 14.40 +
// Windows SDK 22621 + CUDA 12.6 + PyTorch C++ extension pipeline + sm_86).

#include <torch/extension.h>

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

torch::Tensor hello_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.device().is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32, "only float32 supported");
    TORCH_CHECK(b.scalar_type() == torch::kFloat32, "only float32 supported");

    auto c = torch::empty_like(a);
    int n = static_cast<int>(a.numel());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), n);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &hello_add, "Element-wise add on CUDA");
}
