# Copyright 2025 The MuJoCo Warp Authors.
# Licensed under the Apache License, Version 2.0.
"""HIP/CUDA graph capture context manager for mujoco_warp.

Provides hip_graph_capture() -- a portable context manager for
graph capture on both NVIDIA CUDA and AMD ROCm (HIP).

On AMD ROCm, correctly manages mempool state around ScopedCapture
and runs warmup steps to exhaust lazy allocations before capture.
On NVIDIA CUDA, equivalent to wp.ScopedCapture() with warmup.

Example::

    with mjw.hip_graph_capture(model, data) as cap:
        mjw.step(model, data)

    for _ in range(1000):
        wp.capture_launch(cap.graph)  # 1.7x faster on AMD ROCm
"""
from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING
import warp as wp
if TYPE_CHECKING:
    from mujoco_warp._src.types import Data, Model


class _CaptureResult:
    graph = None


@contextlib.contextmanager
def hip_graph_capture(model, data, warmup_steps=3):
    """Portable context manager for CUDA/HIP graph capture of mjw.step().

    Args:
        model: mujoco_warp Model instance.
        data: mujoco_warp Data instance.
        warmup_steps: eager steps before capture to exhaust lazy allocations.

    Yields:
        _CaptureResult with .graph populated after the with-block exits.
    """
    import mujoco_warp as mjw
    device = wp.get_device()
    result = _CaptureResult()

    # Warmup: exhaust all lazy allocations so no hipMallocAsync occurs
    # inside the capture window (which would create memAlloc graph nodes).
    for _ in range(warmup_steps):
        mjw.step(model, data)
    wp.synchronize_device()

    if device.is_hip:
        # ROCm: ScopedCapture requires mempool enabled; restore after.
        pool_was_enabled = wp.is_mempool_enabled(device)
        if not pool_was_enabled:
            wp.set_mempool_enabled(device, True)
        try:
            with wp.ScopedCapture() as capture:
                yield result
        finally:
            if not pool_was_enabled:
                wp.set_mempool_enabled(device, False)
        result.graph = capture.graph
    else:
        with wp.ScopedCapture() as capture:
            yield result
        result.graph = capture.graph
