"""Smoke test: import the compiled extension and verify the CUDA kernel runs."""

import torch
import hello_cuda

assert torch.cuda.is_available(), "CUDA not available — can't run the kernel"

a = torch.randn(1024, device="cuda", dtype=torch.float32)
b = torch.randn(1024, device="cuda", dtype=torch.float32)
c = hello_cuda.add(a, b)

expected = a + b
max_err = (c - expected).abs().max().item()
assert max_err < 1e-5, f"kernel result diverges from torch: max |err| = {max_err}"

print(f"hello_cuda.add works: max |err| = {max_err:.2e}")
print(f"device={c.device}  shape={tuple(c.shape)}  dtype={c.dtype}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
