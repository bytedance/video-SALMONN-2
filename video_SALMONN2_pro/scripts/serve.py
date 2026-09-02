# Copyright (2026) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_plugin_entry_point() -> None:
    """Make `qwenvl.vllm_plugin:register` discoverable as a vLLM plugin.

    vLLM discovers plugins via `importlib.metadata.entry_points(group=...)`,
    which reads installed distribution metadata. Since this project is not
    pip-installed, we drop a minimal `.dist-info` directory (with an
    `entry_points.txt`) into the active site-packages so every spawned vLLM
    process picks up the entry point.
    """
    import site
    import sysconfig

    # Prefer a writable site-packages directory on sys.path.
    candidates = []
    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        candidates.append(purelib)

    target_dir = None
    for cand in candidates:
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            target_dir = cand
            break
    if target_dir is None:
        raise RuntimeError(
            "No writable site-packages directory found to register the vLLM "
            "plugin entry point. Either run with write access to site-packages "
            "or pip-install this project with a `vllm.general_plugins` entry "
            "point declared."
        )

    dist_info = os.path.join(target_dir, "qwenvl_vllm_plugin-0.0.0.dist-info")
    os.makedirs(dist_info, exist_ok=True)

    metadata = (
        "Metadata-Version: 2.1\n"
        "Name: qwenvl-vllm-plugin\n"
        "Version: 0.0.0\n"
    )
    entry_points = (
        "[vllm.general_plugins]\n"
        "qwenvl_qwen3vl = qwenvl.vllm_plugin:register\n"
    )
    with open(os.path.join(dist_info, "METADATA"), "w") as f:
        f.write(metadata)
    with open(os.path.join(dist_info, "entry_points.txt"), "w") as f:
        f.write(entry_points)
    # RECORD may be consulted by some tooling; keep it minimal.
    with open(os.path.join(dist_info, "RECORD"), "w") as f:
        f.write("")

    # Ensure the project root is importable in child processes.
    pythonpath = os.environ.get("PYTHONPATH", "")
    parts = pythonpath.split(os.pathsep) if pythonpath else []
    if str(PROJECT_ROOT) not in parts:
        parts.insert(0, str(PROJECT_ROOT))
        os.environ["PYTHONPATH"] = os.pathsep.join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the project's Qwen3-VL via vLLM's OpenAI API.",
    )
    parser.add_argument(
        "--model",
        default="tsinghua-ee/video-SALMONN-2-Pro-8B",
        help="Model directory (must contain the Qwen3VL config/weights).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-model-name", default="qwen3-vl")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=80000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--limit-video", type=int, default=1)
    parser.add_argument("--limit-audio", type=int, default=1)

    # --- qwen3vl video backend sampling knobs (mirror data_qwen.py) ---
    parser.add_argument("--base-interval", type=float, default=0.1)
    parser.add_argument("--video-min-frames", type=int, default=32)
    parser.add_argument("--video-max-frames", type=int, default=900)
    parser.add_argument("--video-min-frame-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--video-max-frame-pixels", type=int, default=128000)
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable reading the embedded audio track (video-only).",
    )
    parser.add_argument("--audio-sampling-rate", type=int, default=16000)

    # Everything after `--` is forwarded verbatim to `vllm serve`.
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to `vllm serve` (after `--`).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_plugin_entry_point()

    # Force spawn so worker processes reliably re-load the plugin (and so we
    # never fork after CUDA init).
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    # Route video decoding through our custom backend in every process.
    os.environ.setdefault("VLLM_VIDEO_LOADER_BACKEND", "qwen3vl")

    # Sampling parameters are consumed by `Qwen3VLVideoBackend.load_bytes`
    # via `--media-io-kwargs '{"video": {...}}'`. `video_max_frame_pixels` is
    # also forwarded to the HF video processor's pixel budget through the same
    # channel.
    media_io_kwargs = {
        "video": {
            "base_interval": args.base_interval,
            "video_min_frames": args.video_min_frames,
            "video_max_frames": args.video_max_frames,
            "video_min_frame_pixels": args.video_min_frame_pixels,
            "video_max_frame_pixels": args.video_max_frame_pixels,
            "use_audio": not args.no_audio,
            "audio_sampling_rate": args.audio_sampling_rate,
        }
    }

    limit_mm = {"video": args.limit_video, "audio": args.limit_audio}

    # Match data_qwen.process_video_frames pixel budget so the HF video
    # processor produces the same grid_thw / token counts as training:
    #   max_pixels = video_max_frames * video_max_frame_pixels
    #   min_pixels = video_min_frame_pixels
    # These are passed to the HF processor via `mm_processor_kwargs`.
    mm_processor_kwargs = {
        "size": {
            "shortest_edge": args.video_min_frame_pixels,
            "longest_edge": args.video_max_frames * args.video_max_frame_pixels,
        }
    }

    serve_args = [
        "vllm",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--served-model-name",
        args.served_model_name,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--enable-mm-embeds",
        # The custom processor returns fully per-frame-expanded input_ids
        # (interleaved video + audio pads). That layout is only adopted as the
        # prompt on the NON-cached processing path (enable_hf_prompt_update=
        # True). The mm-preprocessor cache would instead re-tokenize the raw
        # text and re-expand via single-block prompt updates, dropping the
        # interleaved audio layout, so we disable it by sizing the cache to 0.
        "--mm-processor-cache-gb",
        "0",
        "--limit-mm-per-prompt",
        json.dumps(limit_mm),
        "--media-io-kwargs",
        json.dumps(media_io_kwargs),
        "--mm-processor-kwargs",
        json.dumps(mm_processor_kwargs),
        "--allowed-local-media-path",
        "/",
    ]

    # Data parallelism: only forward when enabled so single-GPU launches keep
    # vLLM's default behavior untouched.
    if args.data_parallel_size > 1:
        serve_args += ["--data-parallel-size", str(args.data_parallel_size)]

    extra = args.extra
    if extra and extra[0] == "--":
        extra = extra[1:]
    serve_args.extend(extra)

    # Hand off to vLLM's CLI. Use the module entry so we don't depend on the
    # `vllm` console script being on PATH.
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv = serve_args
    vllm_main()


if __name__ == "__main__":
    main()
