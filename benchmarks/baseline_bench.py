"""Baseline Benchmarking on main branch.

Measures the raw performance of the 'main' branch using its native, 
blocking .cpu() and eager-mode paths.
"""

import argparse
import logging
import time
from contextlib import contextmanager

import torch

# Import from main's existing structure
from scope.server.pipeline_manager import PipelineManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("baseline_bench")

class BaselineBenchmarker:
    def __init__(self, pipeline_id: str, frames: int = 50, warmup: int = 10):
        self.pipeline_id = pipeline_id
        self.frames = frames
        self.warmup = warmup
        self.manager = PipelineManager()

    @contextmanager
    def measure(self, label: str, stats: dict):
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000  # ms
        stats[label].append(elapsed)

    def run(self):
        logger.info(f"=== Starting BASELINE Benchmark for {self.pipeline_id} (Branch: main) ===")

        # Load Pipeline (Native main mode: no auto-quant, no compile)
        logger.info("Loading pipeline...")
        # Note: on main, load_params might be different, but empty is safe
        pipeline = self.manager._load_pipeline_implementation(self.pipeline_id, {})
        
        # Warmup
        logger.info(f"Warming up ({self.warmup} frames)...")
        dummy_input = torch.randn(1, 3, 512, 512, device="cuda")
        for _ in range(self.warmup):
            _ = pipeline(video=[dummy_input], init_cache=True)

        # Measurement
        logger.info(f"Measuring ({self.frames} frames)...")
        stats = {"compute_ms": [], "egest_ms": [], "total_ms": []}

        for i in range(self.frames):
            start_total = time.perf_counter()
            
            with self.measure("compute_ms", stats):
                output_dict = pipeline(video=[dummy_input], init_cache=False)
                torch.cuda.synchronize()

            video_out = output_dict.get("video")
            if video_out is not None:
                with self.measure("egest_ms", stats):
                    # Native main behavior: BLOCKING .cpu()
                    cpu_tensor = video_out.cpu()
                    _ = cpu_tensor.numpy()

            stats["total_ms"].append((time.perf_counter() - start_total) * 1000)
            if i % 10 == 0:
                logger.info(f"  Frame {i}/{self.frames}...")

        avg_total = sum(stats["total_ms"]) / len(stats["total_ms"])
        fps = 1000 / avg_total
        
        print("\n" + "="*50)
        print("      MAIN BRANCH BASELINE RESULTS")
        print("="*50)
        print(f"Pipeline:      {self.pipeline_id}")
        print(f"Total Latency: {avg_total:.2f} ms")
        print(f"Throughput:    {fps:.2f} FPS")
        print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="passthrough")
    args = parser.parse_args()
    BaselineBenchmarker(args.pipeline).run()
