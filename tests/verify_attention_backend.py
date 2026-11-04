#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
verify_attention_backend.py -- report which backend PyTorch SDPA actually selects.

torch.nn.functional.scaled_dot_product_attention is a dispatcher that prefers
Flash > Memory-efficient > Math, each with its own dtype restrictions:

    Flash             fp16 / bf16 only   (the official flash-attn kernel)
    Memory-efficient  fp32 supported     (xFormers-style, never materialises L x L)
    Math              fp32 supported     (the plain implementation, full matrix)

Under fp32 the Flash backend is skipped, so it matters which one is actually reached:

  * Memory-efficient: the O(L^2) -> O(L) memory argument holds, and the paper may call
    this "memory-efficient attention";
  * Math: the encoder's attention is no different from a standard implementation, and any
    claim that Flash Attention buys efficiency has to go.

Usage:
    python3 verify_attention_backend.py

Prints, per dtype, which backend is enabled.
"""

import sys

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("torch is not installed; run this in the environment used for training")

print("=" * 74)
print("Environment")
print("=" * 74)
print("  torch           :", torch.__version__)
print("  CUDA (torch)    :", torch.version.cuda)
print("  CUDA available  :", torch.cuda.is_available())

if not torch.cuda.is_available():
    sys.exit("\nNo GPU available; this script needs CUDA (both backends are CUDA kernels)")

device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(device)
print("  GPU             :", props.name)
print("  Compute cap.    : %d.%d" % (props.major, props.minor))
print("  device memory   : %.1f GB" % (props.total_memory / 1024 ** 3))

if props.major < 8:
    print("\n  [warn] compute capability < 8.0 (pre-Ampere): the flash backend is unavailable.")

# ---------------------------------------------------------------------------
# 1) the built-in predicate
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("1) torch.backends.cuda.can_use_flash_attention()")
print("=" * 74)
try:
    print("   returned:", torch.backends.cuda.can_use_flash_attention())
    print("   (judged in a half/bf16 context; it does not answer the fp32 question)")
except AttributeError:
    print("   this API does not exist in your torch version (needs PyTorch >= 2.0)")


# ---------------------------------------------------------------------------
# 2) force one backend at a time and test each dtype
# ---------------------------------------------------------------------------
def try_backend(name, dtype, enable):
    """Enable one backend inside sdp_kernel, run one forward, and report success."""
    B, H, L, D = 1, 8, 1024, 64
    q = torch.randn(B, H, L, D, dtype=dtype, device=device)
    try:
        ctx = torch.nn.attention.sdp_kernel(**enable)
    except AttributeError:
        ctx = torch.backends.cuda.sdp_kernel(**enable)
    try:
        with ctx:
            out = F.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        return True, out.dtype
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, str(e)[:80])


ENABLE = {
    "flash":          dict(enable_flash=True,  enable_math=False, enable_mem_efficient=False),
    "mem_efficient":  dict(enable_flash=False, enable_math=False, enable_mem_efficient=True),
    "math":           dict(enable_flash=False, enable_math=True,  enable_mem_efficient=False),
}
NAMES = {"flash": "Flash", "mem_efficient": "Memory-efficient", "math": "Math"}

print("\n" + "=" * 74)
print("2) backend x dtype, one backend forced at a time")
print("=" * 74)
print("   %-18s %-10s %-10s %-10s" % ("backend", "fp32", "bf16", "fp16"))
results = {}
for backend in ("flash", "mem_efficient", "math"):
    row = []
    for dt, dn in ((torch.float32, "fp32"), (torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        ok, info = try_backend(backend, dt, ENABLE[backend])
        results[(backend, dn)] = ok
        row.append("OK" if ok else "unavailable")
    print("   %-18s %-10s %-10s %-10s" % (NAMES[backend], *row))

# ---------------------------------------------------------------------------
# 3) what does fp32 select when no backend is forced?
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("3) which backend does fp32 select by default?")
print("=" * 74)

# Observe the executed kernel names through the profiler to identify the backend
B, H, L, D = 1, 8, 2048, 64
q = torch.randn(B, H, L, D, dtype=torch.float32, device=device)

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    for _ in range(3):
        F.scaled_dot_product_attention(q, q, q)
    torch.cuda.synchronize()

kernels = set()
for evt in prof.key_averages():
    n = (evt.key or "").lower()
    if any(k in n for k in ("flash", "attention", "efficient", "fmha", "cutlass", "softmax")):
        kernels.add(evt.key)

print("   attention-related kernels observed under fp32:")
for k in sorted(kernels) or ["(no attention kernel captured)"]:
    print("     ·", k)

lowered = " ".join(k.lower() for k in kernels)
verdict = None
if "flash" in lowered or "fmha" in lowered:
    verdict = "flash"
elif "efficient" in lowered or "xformers" in lowered or "cutlass" in lowered:
    verdict = "mem_efficient"
elif kernels:
    verdict = "math"

print("\n" + "=" * 74)
print("Conclusion")
print("=" * 74)
print("   fp32 + flash backend usable     :", results[("flash", "fp32")])
print("   fp32 + mem-efficient usable     :", results[("mem_efficient", "fp32")])
print("   inferred fp32 backend           :", NAMES.get(verdict, "undetermined (see kernel names above)"))
print()
if not results[("flash", "fp32")] and results[("mem_efficient", "fp32")]:
    print("   Training in fp32 skips the flash backend; the memory-efficient backend is used.")
    print("   It never materialises the L x L matrix either, so the memory argument holds.")
elif verdict == "math":
    print("   [warn] Under fp32 the math backend appears to be selected -- plain attention.")
    print("   Any speedup would then come from the encoder swap, not from the kernel.")
else:
    print("   Determine the actual backend from the kernel names printed above.")
print()
print("   Section 2 above also supports a controlled efficiency comparison: forcing math vs")
print("   mem_efficient through sdp_kernel isolates the kernel's time and memory cost under")
print("   one fixed encoder.")
