"""StreamDiffusionV2 Benchmark.
Verifies FPS on the high-speed pipeline variant.
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

with mock.patch("scope.core.pipelines.utils.load_state_dict", side_effect=mock_load_state_dict):
    from scope.core.pipelines.streamdiffusionv2.pipeline import StreamDiffusionV2Pipeline
    from scope.core.pipelines.streamdiffusionv2.schema import StreamDiffusionV2Config
    from scope.core.pipelines.wan2_1.components.generator import WanDiffusionWrapper
    from scope.core.pipelines.wan2_1.components.text_encoder import WanTextEncoderWrapper
    from scope.server.pinned_transfer import gpu_to_cpu

    class MockDiffusionWrapper(WanDiffusionWrapper):
        def __init__(self, causal_model_cls, *args, **kwargs):
            torch.nn.Module.__init__(self)
            with torch.device("cuda"):
                # Use only 2 layers to see if we can hit high FPS
                self.model = causal_model_cls(num_layers=2, dim=1024, num_heads=16, text_len=512, text_dim=4096)
            self.model.eval()
            self.scheduler = mock.Mock()
            self.seq_len = 32760
        def get_scheduler(self): return self.scheduler
        def post_init(self): pass
        def forward(self, noisy_image_or_video, conditional_dict, timestep, **kwargs):
            prompt_embeds = conditional_dict["prompt_embeds"]
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=timestep,
                context=prompt_embeds,
                seq_len=self.seq_len
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
    print(f"--- StreamDiffusionV2 (2-layer Mock) [{mode}] ---")
    
    device = "cuda"
    dtype = torch.bfloat16
    config = StreamDiffusionV2Config(height=320, width=576)
    
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
        "num_layers": 2,
        "text_len": 512
    })

    with mock.patch("scope.core.pipelines.streamdiffusionv2.pipeline.load_model_config", return_value=mocked_cfg):
        with mock.patch("scope.core.pipelines.streamdiffusionv2.pipeline.WanDiffusionWrapper", MockDiffusionWrapper):
            with mock.patch("scope.core.pipelines.streamdiffusionv2.pipeline.StreamDiffusionV2WanVAEWrapper", mock.Mock()):
                pipeline = StreamDiffusionV2Pipeline(config, device=torch.device(device), dtype=dtype)

    # Mock the blocks and output postprocessing
    pipeline.blocks = mock.Mock(return_value=(None, pipeline.state))
    pipeline.state.set("output_video", torch.randn(1, 4, 3, 320, 576, device=device, dtype=dtype))
    
    transfer_fn = gpu_to_cpu if use_turbo else lambda x: x.cpu()
    
    results = {}
    for _ in range(10): # Warmup
        _ = pipeline(prompts=["test"])
    
    start = time.perf_counter()
    for i in range(iterations):
        with timer("compute", results):
            out = pipeline(prompts=["test"])
            video = pipeline.state.get("output_video")
        with timer("egest", results):
            _ = transfer_fn(video)
    
    total = time.perf_counter() - start
    fps = iterations / total
    print(f"  Throughput: {fps:.2f} FPS")
    return fps

if __name__ == "__main__":
    fps_legacy = run_bench(False)
    fps_turbo = run_bench(True)
    print(f"\nGAIN: {(fps_turbo/fps_legacy - 1)*100:+.1f}%")
