# Copyright (2025) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

import os
import logging
import pathlib
import torch
import transformers
import json
from typing import Dict
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
import random
import time

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from qwenvl.model.modeling_qwen2_5_vl import video_SALMONN2_plus
from qwenvl.data.dataset import make_supervised_data_module
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from qwenvl.train.trainer import QwenVLTrainer

from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

from tqdm import tqdm
import torch.distributed as dist

def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen2.5-VL models.
    NOTE: Qwen2.5-VL is not available in transformers<4.48.2

    Args:
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        swiglu (bool): Whether to apply Liger's SwiGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """

    print("Applying Liger kernels to Qwen2.5-VL model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from qwenvl.model import modeling_qwen2_5_vl

    if rope:
        modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = liger_multimodal_rotary_pos_emb
    if rms_norm:
        modeling_qwen2_5_vl.Qwen2RMSNorm = LigerRMSNorm
    if swiglu:
        modeling_qwen2_5_vl.Qwen2MLP = LigerSwiGLUMLP

apply_liger_kernel_to_qwen2_5_vl()

def prepare_inputs(inputs):
    inputs.pop("video", None)
    inputs.pop("image", None)
    inputs.pop("prompt", None)
    inputs.pop("ref", None)
    inputs.pop("audio", None)
    inputs.pop("use_audio", False)
    inputs.pop("should_use", True)
    inputs = {k: v.to(f"cuda:{torch.cuda.current_device()}") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    return inputs


def prepare_dataset(model_path):
    data_args = DataArguments()
    data_args.video_max_frames = 768
    data_args.video_min_frames = 16
    data_args.base_interval = 0.1
    data_args.max_pixels = 61250
    data_args.video_max_frame_pixels = 61250
    data_args.run_test = True
    data_args.image_processor = Qwen2VLImageProcessorFast.from_pretrained(model_path)
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, 
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    data_args.model_type = "qwen2.5vl"

    return data_args



model_path = "tsinghua-ee/video-SALMONN2_plus_3B_full"
data_args = prepare_dataset(model_path)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    model_max_length=131072,
    padding_side="right",
    use_fast=False,
)

model = video_SALMONN2_plus.from_pretrained(
    model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map="cpu"
)
model.cuda()

data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
test_data = data_module["train_dataset"]

input_dict = {
    "video": "video/path",
    "use_audio": True, # False for visual-only input
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nProvide a comprehensive description of all the content in the video, leaving out no details. Be sure to include as much of the audio information as possible, and ensure that your descriptions of the audio and video are closely aligned."
        },
        {
            "from": "gpt",
            "value": ""
        }
    ]
}

inputs = test_data._get_item(input_dict)
inputs = prepare_inputs(inputs)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False
    )
output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
output_text = tokenizer.decode(output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(output_text)
