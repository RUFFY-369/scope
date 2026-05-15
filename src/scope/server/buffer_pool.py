"""Double-buffered pinned memory pool for GPU→CPU frame transfers.

Upgrades the basic `pinned_transfer.gpu_to_cpu()` with a pool-based
approach that supports true double-buffering: while buffer A is being
consumed by downstream (WebRTC H.264 encoding, NDI/Spout send), buffer B
can receive the next GPU frame via async DMA copy.

Usage::

    from .buffer_pool import PinnedBufferPool

    pool = PinnedBufferPool(pool_size=2)

    # Hot path:
    buf = pool.acquire(shape, dtype)
    buf.copy_(gpu_tensor, non_blocking=True)
    torch.cuda.current_stream().synchronize()
    numpy_frame = buf.numpy()
    # ... use numpy_frame ...
    pool.release(buf)
"""

import logging
import threading
from collections import defaultdict

import torch

logger = logging.getLogger(__name__)


class PinnedBufferPool:
    """Thread-safe pool of pinned host memory buffers.

    Pre-allocates ``pool_size`` pinned buffers per (shape, dtype) pair.
    Buffers are reused across frames to eliminate per-frame
    ``torch.empty(..., pin_memory=True)`` allocation overhead.

    When the pool is exhausted (all buffers in-flight), a new buffer is
    allocated on-demand and added to the pool upon release, so the pool
    grows to match peak concurrency but never shrinks.
    """

    def __init__(self, pool_size: int = 2):
        self._pool_size = pool_size
        self._lock = threading.Lock()
        # Key: (shape_tuple, dtype) -> list of available pinned buffers
        self._available: dict[tuple, list[torch.Tensor]] = defaultdict(list)
        self._total_allocated: int = 0

    def acquire(
        self, shape: tuple[int, ...], dtype: torch.dtype = torch.uint8
    ) -> torch.Tensor:
        """Get a pinned buffer from the pool, creating one if needed.

        Args:
            shape: Required tensor shape.
            dtype: Required tensor dtype.

        Returns:
            A pinned CPU tensor ready for ``copy_()`` from GPU.
        """
        key = (shape, dtype)
        with self._lock:
            pool = self._available[key]
            if pool:
                return pool.pop()

        # Pool exhausted or first access — allocate new pinned buffer
        buf = torch.empty(shape, dtype=dtype, pin_memory=True)
        with self._lock:
            self._total_allocated += 1
        return buf

    def release(self, buf: torch.Tensor) -> None:
        """Return a buffer to the pool for reuse.

        Args:
            buf: A previously acquired pinned tensor.
        """
        key = (tuple(buf.shape), buf.dtype)
        with self._lock:
            self._available[key].append(buf)

    def warmup(
        self, shape: tuple[int, ...], dtype: torch.dtype = torch.uint8
    ) -> None:
        """Pre-allocate pool buffers for a known resolution.

        Call at stream start when the output resolution is known to avoid
        allocation stalls on the first few frames.

        Args:
            shape: Expected frame shape (H, W, C) or (B, H, W, C).
            dtype: Expected dtype.
        """
        key = (shape, dtype)
        with self._lock:
            existing = len(self._available[key])
            needed = max(0, self._pool_size - existing)

        for _ in range(needed):
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
            with self._lock:
                self._available[key].append(buf)
                self._total_allocated += 1

        if needed > 0:
            logger.debug(
                "PinnedBufferPool: warmed up %d buffer(s) for shape=%s dtype=%s",
                needed,
                shape,
                dtype,
            )

    @property
    def stats(self) -> dict:
        """Pool statistics for debugging."""
        with self._lock:
            available_count = sum(len(v) for v in self._available.values())
            return {
                "total_allocated": self._total_allocated,
                "available": available_count,
                "in_flight": self._total_allocated - available_count,
                "shapes": {
                    str(k): len(v) for k, v in self._available.items()
                },
            }
