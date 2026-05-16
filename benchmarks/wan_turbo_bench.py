"""Wan 2.1 (LongLive) Turbo Benchmark Suite.

Measures actual FPS on the Wan 2.1 architecture by mocking weights
to fit within hardware constraints (6GB VRAM) while maintaining
the full computational graph.
"""
import time
import torch
import unittest.mock as mock
import logging
import os
import sys
import inspect
from contextlib import contextmanager
from omegaconf import OmegaConf

# Muzzle everything
logging.getLogger("scope").setLevel(logging.ERROR)
os.environ["HF_HUB_OFFLINE"] = "1"

# 1. Monkeypatch huggingface_hub BEFORE any other imports
import huggingface_hub
huggingface_hub.snapshot_download = mock.Mock(return_value="/tmp/mock_path")
huggingface_hub.hf_hub_download = mock.Mock(return_value="/tmp/mock_path")

def mock_load_state_dict(*args, **kwargs):
    return {}

with mock.patch("scope.core.pipelines.utils.load_state_dict", side_effect=mock_load_state_dict):
    from scope.core.pipelines.wan2_1.components.generator import WanDiffusionWrapper
    from scope.core.pipelines.wan2_1.components.text_encoder import WanTextEncoderWrapper
    
    class MockScheduler(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.sigmas = torch.randn(1001, device="cuda")
            self.timesteps = torch.arange(1001, device="cuda")
        def add_noise(self, sample, noise, timesteps):
            return sample + noise

    class MockDiffusionWrapper(WanDiffusionWrapper):
        def __init__(self, causal_model_cls, *args, **kwargs):
            torch.nn.Module.__init__(self)
            with torch.device("cuda"):
                # 24 layers, 1024 dim
                self.model = causal_model_cls(num_layers=24, dim=1024, num_heads=16, text_len=512, text_dim=4096)
            self.model.eval()
            self.model.requires_grad_(False)
            self.scheduler = MockScheduler()
            self.seq_len = 32760
            self.uniform_timestep = False
        def get_scheduler(self): return self.scheduler
        def post_init(self): pass
        def forward(self, noisy_image_or_video, conditional_dict, timestep, **kwargs):
            prompt_embeds = conditional_dict["prompt_embeds"]
            
            sig = inspect.signature(self.model._forward_inference)
            has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            
            if has_var_keyword:
                accepted = kwargs
            else:
                accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}

            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                **accepted
            )
            
            if isinstance(flow_pred, (list, tuple)):
                flow_pred = flow_pred[0]
            
            return flow_pred.permute(0, 2, 1, 3, 4), torch.randn_like(noisy_image_or_video)

    class MockTextEncoderWrapper(WanTextEncoderWrapper):
        def __init__(self, *args, **kwargs):
            torch.nn.Module.__init__(self)
            self.model = mock.Mock()
            self.tokenizer = mock.Mock()
        def __call__(self, *args, **kwargs):
            return {"prompt_embeds": torch.randn(1, 512, 4096, device="cuda", dtype=torch.bfloat16)}

    class MockVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x, *args, **kwargs): return x
        def clear_cache(self): pass
        def encode_to_latent(self, x, *args, **kwargs):
            return torch.randn(x.shape[0], x.shape[2], 16, x.shape[3]//8, x.shape[4]//8, device="cuda", dtype=torch.bfloat16)
        def decode_from_latent(self, x, *args, **kwargs):
            return torch.randn(x.shape[0], 3, x.shape[1], x.shape[3]*8, x.shape[4]*8, device="cuda", dtype=torch.bfloat16)
        def decode_to_pixel(self, x, *args, **kwargs):
            return self.decode_from_latent(x)
        def encode(self, x, *args, **kwargs):
            class MockEncoded:
                def __init__(self, latent): self.latent_dist = mock.Mock(sample=lambda: latent)
            return MockEncoded(self.encode_to_latent(x))
        def decode(self, x, *args, **kwargs):
            class MockDecoded:
                def __init__(self, sample): self.sample = sample
            return MockDecoded(self.decode_from_latent(x))

    # Patch the components
    with mock.patch("scope.core.pipelines.longlive.pipeline.WanDiffusionWrapper", MockDiffusionWrapper):
        with mock.patch("scope.core.pipelines.longlive.pipeline.WanTextEncoderWrapper", MockTextEncoderWrapper):
            with mock.patch("scope.core.pipelines.longlive.pipeline.create_vae", return_value=MockVAE().to("cuda")):
                from scope.core.pipelines.longlive.pipeline import LongLivePipeline
                from scope.core.pipelines.longlive.schema import LongLiveConfig
                from scope.server.pinned_transfer import gpu_to_cpu, gpu_to_cpu_pooled
                
                # Mock try_compile_pipeline to use mode="default"
                def mock_try_compile(pipeline):
                    if not getattr(pipeline, "_compile_eligible", False): return
                    pipeline.__call__ = torch.compile(pipeline.__call__, mode="default", dynamic=False, fullgraph=False)
                    print("  [MOCK] Applied torch.compile (mode=default, dynamic=False)")

                from scope.server import compile_optimizer
                compile_optimizer.try_compile_pipeline = mock_try_compile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wan_turbo_bench")

@contextmanager
def timer(name, results):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    results[name] = results.get(name, 0) + (time.perf_counter() - start)

def run_wan_bench(use_turbo: bool, iterations=10):
    mode = "TURBO" if use_turbo else "LEGACY"
    print(f"\n--- Benchmarking Wan 2.1 Architecture [{mode} Mode] ---")
    
    device = torch.device("cuda")
    dtype = torch.bfloat16
    
    config = LongLiveConfig(height=320, width=576)
    
    mocked_cfg = OmegaConf.create({
        "max_rope_freq_table_seq_len": 1024,
        "sample_fps": 16,
        "sample_video_size": [320, 576],
        "base_model_name": "Wan2.1-T2V-1.3B",
        "num_frame_per_block": 1,
        "independent_first_frame": False,
        "patch_size": [1, 2, 2],
        "vae_spatial_downsample_factor": 8,
        "vae_temporal_downsample_factor": 4,
        "patch_embedding_spatial_downsample_factor": 2,
        "patch_embedding_temporal_downsample_factor": 1,
        "vae_type": "wan",
        "local_attn_size": 12,
        "sink_size": 0,
        "global_sink": True,
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "in_dim": 16,
        "dim": 1024,
        "ffn_dim": 4096,
        "text_dim": 4096,
        "out_dim": 16,
        "num_heads": 16,
        "num_layers": 24,
        "text_len": 512
    })

    with mock.patch("scope.core.pipelines.longlive.pipeline.load_model_config", return_value=mocked_cfg):
        with mock.patch("scope.core.pipelines.longlive.pipeline.WanDiffusionWrapper", MockDiffusionWrapper):
            with mock.patch("scope.core.pipelines.longlive.pipeline.WanTextEncoderWrapper", MockTextEncoderWrapper):
                with mock.patch("scope.core.pipelines.longlive.pipeline.create_vae", return_value=MockVAE().to("cuda")):
                    pipeline = LongLivePipeline(config, device=device, dtype=dtype)

    if use_turbo:
        pipeline._compile_eligible = True
        from scope.server.compile_optimizer import try_compile_pipeline
        try_compile_pipeline(pipeline)
        transfer_fn = gpu_to_cpu
    else:
        transfer_fn = lambda x: x.cpu()

    # Complete inputs
    inputs = {
        "prompts": ["a beautiful video"],
        "negative_prompts": ["ugly, blurry"],
        "video": torch.randn(1, 3, 4, 320, 576, device=device, dtype=dtype),
        "latents": torch.randn(1, 4, 16, 40, 72, device=device, dtype=dtype),
        "prompt_embeds": torch.randn(1, 512, 4096, device=device, dtype=dtype),
        "negative_prompt_embeds": torch.randn(1, 512, 4096, device=device, dtype=dtype),
        "timestep": torch.tensor([[500, 500, 500, 500]], device=device),
        "denoising_step_list": torch.tensor([500, 400], device=device, dtype=torch.long),
        "current_denoising_step_list": torch.tensor([500, 400], device=device, dtype=torch.long),
        "generator": torch.Generator(device=device),
        "current_start_frame": 0,
        "conditioning_embeds_updated": True,
        "kv_cache": None,
        "crossattn_cache": None,
        "kv_bank": None
    }

    results = {}
    warmup = 15
    
    print(f"  Warmup ({warmup} frames)...")
    for _ in range(warmup):
        _ = pipeline._generate(**inputs)
    
    torch.cuda.synchronize()
    
    print(f"  Measuring {iterations} frames...")
    start_total = time.perf_counter()
    for i in range(iterations):
        with timer("compute", results):
            output_dict = pipeline._generate(**inputs)
            video_tensor = output_dict["video"]
        
        with timer("egest", results):
            _ = transfer_fn(video_tensor)
            
    torch.cuda.synchronize()
    end_total = time.perf_counter()
    
    total_time = end_total - start_total
    fps = iterations / total_time
    
    print(f"\n[{mode} RESULTS]")
    print(f"  Throughput: {fps:.2f} FPS")
    print(f"  Avg Compute: {results.get('compute', 0)/iterations*1000:.2f} ms")
    print(f"  Avg Egest:   {results.get('egest', 0)/iterations*1000:.2f} ms")
    
    return fps

if __name__ == "__main__":
    torch.cuda.empty_cache()
    fps_legacy = run_wan_bench(use_turbo=False, iterations=30) # Reduced slightly for time
    torch.cuda.empty_cache()
    fps_turbo = run_wan_bench(use_turbo=True, iterations=30)
    
    gain = (fps_turbo / fps_legacy - 1) * 100
    print(f"\n==================================================")
    print(f"FINAL WAN 2.1 ARCHITECTURE GAIN: {gain:+.1f}%")
    print(f"==================================================")
