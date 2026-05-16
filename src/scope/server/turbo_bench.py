"""Scope Turbo Benchmarking Suite.

Provides an A/B testing framework to measure the impact of Zero-Copy DMA,
torch.compile, and Auto-Quantization on the running Scope engine.

Usage:
    uv run python -m scope.server.turbo_bench --pipeline streamdiffusionv2
"""

import argparse
import logging
import time
from contextlib import contextmanager

import torch

from .hardware_optimizer import get_gpu_profile
from .pipeline_manager import PipelineManager
from .pinned_transfer import gpu_to_cpu

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("turbo_bench")


class Benchmarker:
    def __init__(self, pipeline_id: str, frames: int = 50, warmup: int = 10):
        self.pipeline_id = pipeline_id
        self.frames = frames
        self.warmup = warmup
        self.manager = PipelineManager()
        self.gpu_profile = get_gpu_profile()

    @contextmanager
    def measure(self, label: str, stats: dict):
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000  # ms
        stats[label].append(elapsed)

    def run_benchmark(self, use_turbo: bool):
        mode = "TURBO" if use_turbo else "LEGACY"
        logger.info(f"=== Starting {mode} Benchmark for {self.pipeline_id} ===")

        # 1. Setup load parameters
        load_params = {
            "quantization": None if not use_turbo else "auto" # 'auto' triggers our hardware optimizer
        }
        
        # We manually control compile to show the difference
        import os
        if not use_turbo:
            os.environ["SCOPE_DISABLE_COMPILE"] = "1"
        else:
            os.environ.pop("SCOPE_DISABLE_COMPILE", None)

        # 2. Load Pipeline
        logger.info(f"[{mode}] Loading pipeline...")
        pipeline = self.manager._load_pipeline_implementation(self.pipeline_id, load_params)
        
        # Phase 3 Hook
        if use_turbo:
            from .compile_optimizer import try_compile_pipeline
            try_compile_pipeline(pipeline)

        # 3. Warmup
        logger.info(f"[{mode}] Warming up ({self.warmup} frames)...")
        # Dummy inputs based on pipeline needs (simplified for bench)
        dummy_input = torch.randn(1, 3, 512, 512, device="cuda")
        for _ in range(self.warmup):
            _ = pipeline(video=[dummy_input], init_cache=True)

        # 4. Measurement
        logger.info(f"[{mode}] Measuring ({self.frames} frames)...")
        stats = {
            "compute_ms": [],
            "egest_ms": [],
            "total_ms": []
        }

        for i in range(self.frames):
            start_total = time.perf_counter()
            
            # Compute step
            with self.measure("compute_ms", stats):
                output_dict = pipeline(video=[dummy_input], init_cache=False)
                torch.cuda.synchronize() # Ensure compute is done

            # Egest step (GPU -> CPU)
            video_out = output_dict.get("video")
            if video_out is not None:
                with self.measure("egest_ms", stats):
                    if use_turbo:
                        # Phase 1: Async pinned transfer
                        cpu_tensor = gpu_to_cpu(video_out)
                        _ = cpu_tensor.numpy()
                    else:
                        # Legacy: Blocking .cpu()
                        cpu_tensor = video_out.cpu()
                        _ = cpu_tensor.numpy()

            stats["total_ms"].append((time.perf_counter() - start_total) * 1000)
            if i % 10 == 0:
                logger.info(f"  Frame {i}/{self.frames}...")

        # 5. Summary
        avg_compute = sum(stats["compute_ms"]) / len(stats["compute_ms"])
        avg_egest = sum(stats["egest_ms"]) / len(stats["egest_ms"])
        avg_total = sum(stats["total_ms"]) / len(stats["total_ms"])
        fps = 1000 / avg_total
        
        vram = torch.cuda.max_memory_allocated() / (1024**2)

        return {
            "mode": mode,
            "avg_compute_ms": avg_compute,
            "avg_egest_ms": avg_egest,
            "avg_total_ms": avg_total,
            "fps": fps,
            "vram_mb": vram
        }

def print_results(results_legacy, results_turbo):
    print("\n" + "="*50)
    print("      SCOPE TURBO BENCHMARK RESULTS")
    print("="*50)
    print(f"{'Metric':<20} | {'Legacy':<12} | {'Turbo':<12} | {'Gain'}")
    print("-" * 60)
    
    def row(label, legacy, turbo, is_fps=False):
        gain = (turbo / legacy - 1) * 100 if is_fps else (1 - turbo / legacy) * 100
        print(f"{label:<20} | {legacy:>10.2f} | {turbo:>10.2f} | {gain:>+6.1f}%")

    row("Compute (ms)", results_legacy["avg_compute_ms"], results_turbo["avg_compute_ms"])
    row("Egest (ms)", results_legacy["avg_egest_ms"], results_turbo["avg_egest_ms"])
    row("Total Latency (ms)", results_legacy["avg_total_ms"], results_turbo["avg_total_ms"])
    row("Throughput (FPS)", results_legacy["fps"], results_turbo["fps"], is_fps=True)
    row("VRAM Usage (MB)", results_legacy["vram_mb"], results_turbo["vram_mb"])
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="streamdiffusionv2")
    args = parser.parse_args()

    bench = Benchmarker(args.pipeline)
    
    # Run Legacy
    results_legacy = bench.run_benchmark(use_turbo=False)
    
    # Clear cache for fair test
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Run Turbo
    results_turbo = bench.run_benchmark(use_turbo=True)
    
    print_results(results_legacy, results_turbo)
