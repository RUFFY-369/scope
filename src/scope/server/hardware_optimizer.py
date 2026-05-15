"""Hardware-aware quantization and precision utilities.

Detects GPU architecture and recommends optimal precision settings
for the Livepeer fleet (RTX 3090 / RTX 4090).

Key capabilities:
- GPU architecture detection (Ampere vs Ada Lovelace)
- FP8 eligibility check (Ada Lovelace only — native FP8 tensor cores)
- Auto-quantization recommendation based on VRAM and architecture
- Integration with existing ``Quantization.FP8_E4M3FN`` enum

Usage::

    from .hardware_optimizer import get_gpu_profile, recommend_quantization

    profile = get_gpu_profile()
    # GPUProfile(arch='ada_lovelace', compute_cap=(8, 9), vram_gb=24.0,
    #            supports_fp8=True, name='NVIDIA GeForce RTX 4090')

    quant = recommend_quantization(profile, pipeline_vram_threshold=20.0)
    # 'fp8_e4m3fn' on 4090, None on 3090 with >24GB
"""

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


# GPU architecture names based on compute capability
_ARCH_NAMES = {
    (7, 0): "volta",
    (7, 5): "turing",
    (8, 0): "ampere",
    (8, 6): "ampere",  # GA106 (RTX 3060, 3070, 3080)
    (8, 9): "ada_lovelace",  # AD102/AD103/AD104 (RTX 4090, 4080, 4070)
    (9, 0): "hopper",
    (10, 0): "blackwell",
}


@dataclass
class GPUProfile:
    """Detected GPU hardware profile."""

    arch: str
    """Architecture name (e.g. 'ampere', 'ada_lovelace')."""

    compute_cap: tuple[int, int]
    """CUDA compute capability (major, minor)."""

    vram_gb: float
    """Total VRAM in GB."""

    supports_fp8: bool
    """Whether the GPU has native FP8 tensor core support."""

    supports_bf16: bool
    """Whether the GPU has native BF16 tensor core support."""

    name: str
    """GPU device name string."""


def get_gpu_profile(device_index: int = 0) -> GPUProfile | None:
    """Detect the GPU profile for the given device.

    Args:
        device_index: CUDA device index.

    Returns:
        GPUProfile if a CUDA GPU is available, None otherwise.
    """
    if not torch.cuda.is_available():
        return None

    try:
        major, minor = torch.cuda.get_device_capability(device_index)
        compute_cap = (major, minor)

        _, total_mem = torch.cuda.mem_get_info(device_index)
        vram_gb = total_mem / (1024**3)

        name = torch.cuda.get_device_name(device_index)

        # FP8 native support: Ada Lovelace (8.9) and newer
        supports_fp8 = major > 8 or (major == 8 and minor >= 9)

        # BF16 native support: Ampere (8.0) and newer
        supports_bf16 = major >= 8

        arch = _ARCH_NAMES.get(compute_cap, f"unknown_{major}_{minor}")

        profile = GPUProfile(
            arch=arch,
            compute_cap=compute_cap,
            vram_gb=round(vram_gb, 1),
            supports_fp8=supports_fp8,
            supports_bf16=supports_bf16,
            name=name,
        )

        logger.info(
            "GPU profile: %s (%s), %.1f GB VRAM, FP8=%s, BF16=%s",
            name,
            arch,
            vram_gb,
            supports_fp8,
            supports_bf16,
        )

        return profile

    except Exception as e:
        logger.warning("Failed to detect GPU profile: %s", e)
        return None


def recommend_quantization(
    profile: GPUProfile | None,
    pipeline_vram_threshold: float | None = None,
) -> str | None:
    """Recommend a quantization method based on hardware profile.

    Decision logic:
    1. If GPU supports FP8 (Ada Lovelace / 4090): recommend FP8.
    2. If GPU is Ampere (3090) and VRAM is below threshold: recommend FP8
       (software-emulated, still saves VRAM).
    3. Otherwise: no quantization (full precision is fine).

    Args:
        profile: GPU hardware profile from ``get_gpu_profile()``.
        pipeline_vram_threshold: VRAM threshold from pipeline config
            (``recommended_quantization_vram_threshold``). If the GPU
            has more VRAM than this, quantization is not recommended.

    Returns:
        Quantization method string (e.g. 'fp8_e4m3fn') or None.
    """
    if profile is None:
        return None

    # If VRAM exceeds the pipeline's recommendation threshold, skip quantization
    if pipeline_vram_threshold is not None:
        if profile.vram_gb > pipeline_vram_threshold:
            return None

    # Ada Lovelace (RTX 4090): native FP8 tensor cores → always recommend
    if profile.supports_fp8:
        return "fp8_e4m3fn"

    # Ampere (RTX 3090): FP8 is software-emulated but still saves VRAM
    if profile.arch == "ampere" and profile.vram_gb <= 24.0:
        return "fp8_e4m3fn"

    return None


def get_optimal_dtype(profile: GPUProfile | None) -> torch.dtype:
    """Get the optimal compute dtype for the detected GPU.

    Args:
        profile: GPU hardware profile.

    Returns:
        torch.bfloat16 for Ampere+, torch.float16 for older GPUs.
    """
    if profile is None:
        return torch.float16

    if profile.supports_bf16:
        return torch.bfloat16

    return torch.float16
