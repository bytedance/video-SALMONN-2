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

"""Run single-video inference with the local video-SALMONN 2 model."""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.data.data_qwen import LazySupervisedDataset
from qwenvl.data.processing_qwen3_vl import Qwen3VLProcessor
from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.train.argument import DataArguments


DEFAULT_PROMPT = (
    "Provide a comprehensive description of all the content in the video, "
    "leaving out no details. Be sure to include as much of the audio "
    "information as possible, and ensure that your descriptions of the audio "
    "and video are closely aligned."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a response for one local video."
    )
    parser.add_argument("--model", default="tsinghua-ee/video-SALMONN-2-Pro-8B")
    parser.add_argument(
        "--processor",
        default=None,
        help="Processor/tokenizer directory. Defaults to --model.",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--no-audio", action="store_true")

    parser.add_argument("--video-min-frames", type=int, default=16)
    parser.add_argument("--video-max-frames", type=int, default=900)
    parser.add_argument("--base-interval", type=float, default=0.1)
    parser.add_argument("--min-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=128000)
    parser.add_argument("--video-min-frame-pixels", type=int, default=4 * 28 * 28)
    parser.add_argument("--video-max-frame-pixels", type=int, default=128000)
    parser.add_argument("--model-max-length", type=int, default=131072)

    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device such as cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
    )
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def build_dataset(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    processor: Qwen3VLProcessor,
) -> LazySupervisedDataset:
    data_args = DataArguments(
        video_min_frames=args.video_min_frames,
        video_max_frames=args.video_max_frames,
        base_interval=args.base_interval,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        video_min_frame_pixels=args.video_min_frame_pixels,
        video_max_frame_pixels=args.video_max_frame_pixels,
    )
    data_args.video_processor = processor.video_processor
    data_args.image_processor = processor.image_processor
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size,
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    return LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)


def prepare_inputs(
    dataset: LazySupervisedDataset,
    video: str,
    prompt: str,
    use_audio: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample = {
        "video": video,
        "use_audio": use_audio,
        "conversations": [
            {
                "from": "human",
                "value": f"<video>\n{prompt}",
            },
            {
                "from": "gpt",
                "value": "",
            },
        ],
    }
    inputs = dataset._get_item([sample], inference=True)
    return {
        key: value.to(device)
        for key, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("--temperature must be positive when --do-sample is set")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    processor_path = args.processor or args.model

    processor = Qwen3VLProcessor.from_pretrained(processor_path)
    tokenizer = AutoTokenizer.from_pretrained(
        processor_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    dataset = build_dataset(args, tokenizer, processor)
    inputs = prepare_inputs(
        dataset,
        str(video_path),
        args.prompt,
        not args.no_audio,
        device,
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        attn_implementation=args.attn_implementation,
        torch_dtype=dtype,
        device_map="cpu",
    )
    model.to(device)
    model.eval()

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    output_text = tokenizer.decode(
        outputs[0, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text.strip())


if __name__ == "__main__":
    main()
