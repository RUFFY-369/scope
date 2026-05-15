"""Compile optimization for pipeline inference.

Wraps a pipeline's ``__call__`` method with ``torch.compile`` when the
pipeline declares itself as compile-eligible and the hardware supports it.

Integration point: called from ``_load_pipeline_implementation`` after
a pipeline instance is created, before it is returned to the caller.

Why this module exists instead of compiling inside each pipeline:

1. **Consistency** — one place controls the compile mode, backend, and
   error handling for all pipelines (built-in and plugin).
2. **Hardware-awareness** — compile mode is chosen based on GPU arch
   (e.g. ``reduce-overhead`` for Ampere+, ``default`` for older GPUs).
3. **Safety** — compilation errors are caught here so a failing compile
   never blocks pipeline loading. The pipeline falls back to eager mode.
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# Environment variable to force-disable compile optimization.
# Useful for debugging or when torch.compile causes issues.
_COMPILE_DISABLED = os.environ.get("SCOPE_DISABLE_COMPILE", "").lower() in (
    "1",
    "true",
    "yes",
)


def get_compile_mode() -> str | None:
    """Select the optimal torch.compile mode for the current GPU.

    Returns:
        Compile mode string, or None if compile should be skipped.
    """
    if _COMPILE_DISABLED:
        logger.info("torch.compile disabled by SCOPE_DISABLE_COMPILE env var")
        return None

    if not torch.cuda.is_available():
        return None

    # torch.compile with reduce-overhead (CUDA Graphs backend) requires
    # compute capability >= 7.0 (Volta+). For older GPUs, use 'default'.
    try:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            # Ampere (3090) and Ada Lovelace (4090): use reduce-overhead
            # which internally uses CUDA Graphs for individual subgraphs
            return "reduce-overhead"
        elif major >= 7:
            return "default"
        else:
            logger.info(
                "GPU compute capability %d.%d too old for torch.compile",
                major,
                minor,
            )
            return None
    except Exception:
        return "default"


def try_compile_pipeline(pipeline) -> None:
    """Attempt to compile a pipeline's __call__ method in-place.

    If the pipeline has a ``_compile_eligible`` attribute set to True,
    wraps ``pipeline.__call__`` with ``torch.compile``. The compiled
    version replaces the original method on the instance.

    If compilation fails for any reason, logs a warning and leaves the
    pipeline unmodified (eager mode fallback).

    Args:
        pipeline: A pipeline instance (any object with ``__call__``).
    """
    # Check eligibility flag
    if not getattr(pipeline, "_compile_eligible", False):
        return

    mode = get_compile_mode()
    if mode is None:
        return

    pipeline_name = getattr(pipeline, "pipeline_id", type(pipeline).__name__)

    try:
        original_call = pipeline.__call__

        compiled_call = torch.compile(
            original_call,
            mode=mode,
            dynamic=True,  # Handle resolution changes without full recompile
            fullgraph=False,  # Allow graph breaks (safer for complex pipelines)
        )

        # Replace __call__ on the instance (not the class)
        pipeline.__call__ = compiled_call

        logger.info(
            "torch.compile applied to %s (mode=%s, dynamic=True)",
            pipeline_name,
            mode,
        )

    except Exception as e:
        logger.warning(
            "torch.compile failed for %s, running in eager mode: %s",
            pipeline_name,
            e,
        )
