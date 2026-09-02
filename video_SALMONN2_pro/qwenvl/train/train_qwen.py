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
import os
import logging
import pathlib
import torch
import transformers
from importlib import import_module
import sys
from pathlib import Path
import numpy as np
import random

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.model.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from qwenvl.data.data_qwen import make_supervised_data_module
from qwenvl.data.processing_qwen3_vl import Qwen3VLProcessor
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoTokenizer, WhisperFeatureExtractor
from qwenvl.train.trainer import QwenVLTrainer

from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
# from liger_kernel.transformers.layer_norm import LigerLayerNorm
# from liger_kernel.transformers.rope import (
#     liger_rotary_pos_emb,
#     liger_rotary_pos_emb_vision
# )

import torch.distributed as dist

import_module("qwenvl.train.deepspeed_patch")

def apply_liger_kernel_to_qwen2_5_vl(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    layer_norm: bool = True
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

    print("Applying Liger kernels to Qwen3-VL model...")

    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from qwenvl.model import modeling_qwen3_vl
    # if rope:
    #     modeling_qwen3_vl.apply_rotary_pos_emb = liger_rotary_pos_emb
    #     modeling_qwen3_vl.apply_rotary_pos_emb_vision = liger_rotary_pos_emb_vision
    if rms_norm:
        modeling_qwen3_vl.Qwen3VLTextRMSNorm = LigerRMSNorm
    # if layer_norm:
    #     modeling_qwen3_vl.nn.LayerNorm = LigerLayerNorm
    if swiglu:
        modeling_qwen3_vl.Qwen3VLTextMLP = LigerSwiGLUMLP


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        model.model.visual.requires_grad_(True)
    else:
        model.model.visual.requires_grad_(False)

    if model_args.tune_mm_mlp:
        model.model.visual.merger.requires_grad_(True)
    else:
        model.model.visual.merger.requires_grad_(False)

    if model_args.tune_mm_audio:
        model.model.audio.requires_grad_(True)
    else:
        model.model.audio.requires_grad_(False)

    if model_args.tune_mm_qformer:
        model.model.audio.qformer.requires_grad_(True)
        model.model.audio.q_tokens.requires_grad_(True)
        model.model.audio.audio_proj.requires_grad_(True)
    else:
        model.model.audio.qformer.requires_grad_(False)
        model.model.audio.q_tokens.requires_grad_(False)
        model.model.audio.audio_proj.requires_grad_(False)

    if model_args.tune_mm_llm:
        if model_args.use_lora:
            raise Exception("tune_mm_llm is not supported when use_lora is True")
        model.model.language_model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    else:
        model.model.language_model.requires_grad_(False)
        model.lm_head.requires_grad_(False)


def train(attn_implementation="flash_attention_3"):
    print("Start")

    seed = 2025
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.remove_unused_columns = False

    apply_liger_kernel_to_qwen2_5_vl()

    os.makedirs(training_args.output_dir, exist_ok=True)

    processor = Qwen3VLProcessor.from_pretrained(
        model_args.model_base,
        cache_dir=training_args.cache_dir,
    )
    data_args.video_processor = processor.video_processor
    data_args.image_processor = processor.image_processor
    data_args.audio_processor = WhisperFeatureExtractor(
        feature_size=data_args.feature_size, 
        sampling_rate=data_args.sampling_rate,
        hop_length=data_args.hop_length,
        chunk_length=data_args.chunk_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_base,
        cache_dir=training_args.cache_dir,
        padding_side="right",
        use_fast=False,
    )
    processor.tokenizer = tokenizer

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # time.sleep(random.randint(0, 20))
    # print(f"RANK {dist.get_rank()} before barrier")
    dist.barrier(device_ids=dist.get_rank())
    # print(f"RANK {dist.get_rank()} after barrier")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_base,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
    )
        
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        if training_args.deepspeed is not None and "3" not in training_args.deepspeed:
            if training_args.gradient_checkpointing_kwargs is None:
                training_args.gradient_checkpointing_kwargs={"use_reentrant": False}
            else:
                training_args.gradient_checkpointing_kwargs["use_reentrant"] = False

    set_model(model_args, model)

    if training_args.no_audio:
        del model.model.audio

    if model_args.use_lora:
        from peft import LoraConfig, get_peft_model
        modules_to_save = []
        if model_args.tune_mm_vision:
            modules_to_save.append("visual")
        if model_args.tune_mm_mlp:
            modules_to_save.append("visual.merger")
        if model_args.tune_mm_audio:
            modules_to_save.append("audio")
        if model_args.tune_mm_qformer:
            modules_to_save.append("audio.qformer")
            modules_to_save.append("audio.q_tokens")
            modules_to_save.append("audio.audio_proj")
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj"], # find_all_linear_names(model),
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=modules_to_save,
        )
        if not training_args.no_audio:
            audio_layers = model.model.audio.layers
            del model.model.audio.layers
        model = get_peft_model(model, lora_config)
        if not training_args.no_audio:
            model.model.model.audio.layers = audio_layers

        for k, v in model.named_parameters():
            if "lora" in k:
                v.requires_grad_(True)
    
    if dist.get_rank() == 0:
        for k, v in model.named_parameters():
            if v.requires_grad:
                print(k, v.shape)
        # print(model.model.visual.merger)

    # print(model.name_or_path)
    trainer = QwenVLTrainer(
        model=model, processing_class=tokenizer, args=training_args, **data_module
    )
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if training_args.should_save:
        processor.save_pretrained(training_args.output_dir)

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    train(attn_implementation="flash_attention_3")
