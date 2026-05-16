"""Synthetic High-Load Benchmarker.
Simulates a heavy diffusion model without requiring model downloads.
Tests Turbo Engine under high thermal and compute pressure.
"""
import time
import torch
from scope.server.pinned_transfer import gpu_to_cpu

def run_blast_test(use_turbo: bool, iterations=10, size=512):
    mode = "TURBO" if use_turbo else "LEGACY"
    print(f"--- Starting {mode} High-Load Blast (Simulating UNet) ---")
    
    # Pre-allocate tensors
    a = torch.randn(size, size, device="cuda")
    b = torch.randn(size, size, device="cuda")
    
    warmup = 10
    measured = 50
    
    # Warmup
    for _ in range(warmup):
        for _ in range(iterations):
            _ = torch.matmul(a, b)
        torch.cuda.synchronize()

    # Measurement
    start = time.perf_counter()
    for i in range(measured):
        # 1. Simulate High Compute
        for _ in range(iterations):
            _ = torch.matmul(a, b)
        
        # 2. Simulate Video Output
        video_out = torch.randn(1, 3, 512, 512, device="cuda")
        
        if use_turbo:
            # Phase 1/2: Async Pinned Transfer
            cpu_tensor = gpu_to_cpu(video_out)
            _ = cpu_tensor.numpy()
        else:
            # Legacy: Blocking .cpu()
            cpu_tensor = video_out.cpu()
            _ = cpu_tensor.numpy()
            
        if i % 10 == 0:
            print(f"  Frame {i}/{measured}...")

    end = time.perf_counter()
    fps = measured / (end - start)
    print(f"{mode} Throughput: {fps:.2f} FPS")
    return fps

if __name__ == "__main__":
    print("Pre-warming GPU...")
    # Run Legacy
    fps_legacy = run_blast_test(use_turbo=False)
    # Run Turbo
    fps_turbo = run_blast_test(use_turbo=True)
    
    gain = (fps_turbo / fps_legacy - 1) * 100
    print(f"\nFINAL LOAD-TEST GAIN: {gain:+.1f}%")
