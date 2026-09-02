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

"""Merge a trained LoRA adapter into a dense base model."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a video-SALMONN 2 LoRA adapter into its base model."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument(
        "--model_base",
        default=None,
        help="Tokenizer and processor source. Defaults to --model_name_or_path.",
    )
    parser.add_argument("--lora_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--no_audio", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPES),
        default="bfloat16",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_base = args.model_base or args.model_name_or_path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_base,
        padding_side="right",
        use_fast=False,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=DTYPES[args.dtype],
        device_map="cpu",
    )

    audio_layers = None
    if not args.no_audio:
        audio_layers = model.model.audio.layers
        del model.model.audio.layers

    from peft import PeftConfig, PeftModel

    model = PeftModel.from_pretrained(model, args.lora_ckpt)

    if audio_layers is not None:
        model.model.model.audio.layers = audio_layers

    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(Path(model_base) / "preprocessor_config.json", "r") as f:
        preprocessor_config = json.load(f)
    preprocessor_config["size"]["longest_edge"] = 176400
    preprocessor_config["size"]["shortest_edge"] = 784
    with open(output_dir / "preprocessor_config.json", "w") as f:
        json.dump(preprocessor_config, f, indent=4)

    with open(Path(model_base) / "video_preprocessor_config.json", "r") as f:
        video_preprocessor_config = json.load(f)
    video_preprocessor_config["size"]["longest_edge"] = (
        163840000
    )
    video_preprocessor_config["size"]["shortest_edge"] = (
        16384
    )
    with open(output_dir / "video_preprocessor_config.json", "w") as f:
        json.dump(video_preprocessor_config, f, indent=4)

    print(output_dir)


if __name__ == "__main__":
    main()
