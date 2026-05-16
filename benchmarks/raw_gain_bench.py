"""Raw Engine Gain Benchmark.
Isolates Turbo Engine optimizations by using a 1-layer model.
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

logging.getLogger("scope").setLevel(logging.ERROR)
os.environ["HF_HUB_OFFLINE"] = "1"

import huggingface_hub
huggingface_hub.snapshot_download = mock.Mock(return_value="/tmp/mock_path")
huggingface_hub.hf_hub_download = mock.Mock(return_value="/tmp/mock_path")

def mock_load_state_dict(*args, **kwargs): return {}

class MockTextEncoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = mock.Mock()
    def __call__(self, *args, **kwargs):
        return {"prompt_embeds": torch.randn(1, 512, 4096, device="cuda", dtype=torch.bfloat16)}
    def to(self, *args, **kwargs): return self

class MockVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def encode_to_latent(self, x, *args, **kwargs):
        return torch.randn(x.shape[0], x.shape[2], 16, x.shape[3]//8, x.shape[4]//8, device="cuda", dtype=torch.bfloat16)
    def decode_from_latent(self, x, *args, **kwargs):
        return torch.randn(x.shape[0], 3, x.shape[1], x.shape[3]*8, x.shape[4]*8, device="cuda", dtype=torch.bfloat16)
    def clear_cache(self): pass
    def to(self, *args, **kwargs): return self

with mock.patch("scope.core.pipelines.utils.load_state_dict", side_effect=mock_load_state_dict):
    with mock.patch("scope.core.pipelines.longlive.pipeline.WanTextEncoderWrapper", MockTextEncoder):
        from scope.core.pipelines.longlive.pipeline import LongLivePipeline
        from scope.core.pipelines.longlive.schema import LongLiveConfig
        from scope.core.pipelines.wan2_1.components.generator import WanDiffusionWrapper
        from scope.server.pinned_transfer import gpu_to_cpu

    class MockDiffusionWrapper(WanDiffusionWrapper):
        def __init__(self, causal_model_cls, *args, **kwargs):
            torch.nn.Module.__init__(self)
            with torch.device("cuda"):
                self.model = causal_model_cls(num_layers=1, dim=1024, num_heads=16, text_len=512, text_dim=4096)
            self.model.eval()
            self.scheduler = mock.Mock()
            self.seq_len = 32760
        def get_scheduler(self): return self.scheduler
        def post_init(self): pass
        def forward(self, noisy_image_or_video, conditional_dict, timestep, **kwargs):
            prompt_embeds = conditional_dict["prompt_embeds"]
            kv_cache = kwargs.get("kv_cache")
            crossattn_cache = kwargs.get("crossattn_cache")
            flow_pred = self.model._forward_inference(
                noisy_image_or_video.permute(0, 2, 1, 3, 4), 
                t=timestep, 
                context=[p for p in prompt_embeds], 
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache
            )
            if isinstance(flow_pred, (list, tuple)): flow_pred = flow_pred[0]
            return flow_pred.permute(0, 2, 1, 3, 4), torch.randn_like(noisy_image_or_video)

@contextmanager
def timer(name, results):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    results[name] = results.get(name, 0) + (time.perf_counter() - start)

def run_bench(use_turbo: bool, iterations=50):
    mode = "TURBO" if use_turbo else "LEGACY"
    print(f"--- Raw Engine Test (1-layer) [{mode}] ---")
    
    device = "cuda"
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
        "num_layers": 1,
        "text_len": 512
    })

    with mock.patch("scope.core.pipelines.longlive.pipeline.load_model_config", return_value=mocked_cfg):
        with mock.patch("scope.core.pipelines.longlive.pipeline.WanDiffusionWrapper", MockDiffusionWrapper):
            with mock.patch("scope.core.pipelines.longlive.pipeline.WanTextEncoderWrapper", MockTextEncoder):
                with mock.patch("scope.core.pipelines.longlive.pipeline.create_vae", return_value=MockVAE().to("cuda")):
                    pipeline = LongLivePipeline(config, device=torch.device(device), dtype=dtype)

    if use_turbo:
        transfer_fn = gpu_to_cpu
    else:
        transfer_fn = lambda x: x.cpu()

    # Initial KV cache
    kv_cache = {
        0: {
            "k": torch.zeros(1, 32760, 16, 64, device=device, dtype=dtype),
            "v": torch.zeros(1, 32760, 16, 64, device=device, dtype=dtype),
            "global_end_index": torch.tensor([0], device=device),
            "local_end_index": torch.tensor([0], device=device)
        }
    }
    crossattn_cache = {0: None}

    # Inputs
    inputs = {
        "prompts": ["test"],
        "video": torch.randn(1, 3, 4, 320, 576, device=device, dtype=dtype),
        "latents": torch.randn(1, 4, 16, 40, 72, device=device, dtype=dtype),
        "prompt_embeds": torch.randn(1, 512, 4096, device=device, dtype=dtype),
        "timestep": torch.tensor([[500, 500, 500, 500]], device=device),
        "denoising_step_list": [500],
        "generator": torch.Generator(device=device),
        "current_start_frame": 0,
        "kv_cache": kv_cache,
        "crossattn_cache": crossattn_cache
    }

    results = {}
    for _ in range(10): _ = pipeline._generate(**inputs) # Warmup
    
    start = time.perf_counter()
    for i in range(iterations):
        with timer("compute", results):
            out = pipeline._generate(**inputs)
            video = out["video"]
        with timer("egest", results):
            _ = transfer_fn(video)
    
    total = time.perf_counter() - start
    fps = iterations / total
    print(f"  Throughput: {fps:.2f} FPS")
    return fps

if __name__ == "__main__":
    fps_legacy = run_bench(False)
    fps_turbo = run_bench(True)
    print(f"\nRAW ARCHITECTURAL GAIN: {(fps_turbo/fps_legacy - 1)*100:+.1f}%")
