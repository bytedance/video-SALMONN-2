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

# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.
# Adopted from https://github.com/huggingface/transformers. The original license is located at 'third-party-license/transformers.txt'.

import copy
import os
import sys

import tqdm
import wandb
import logging
import tempfile
import shutil
import traceback
from pathlib import Path
from transformers import TrainerCallback
from transformers.utils import logging as transformers_logging

# Add parent directory to path for llava imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))
from llava import conversation as conversation_lib
from llava.mm_utils import KeywordsStoppingCriteria
from qwenvl.data.dataset import LazySupervisedDataset, make_supervised_data_module

logger = transformers_logging.get_logger(__name__)
# from contextlib import contextmanager, nullcontext

from transformers.models.auto.modeling_auto import (
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)

import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_varlen_func
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer
from transformers.cache_utils import Cache
from qwenvl.model.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    get_parameter_names,
    has_length,
    is_sagemaker_mp_enabled,
)
import torch.distributed as dist

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    SaveStrategy,
)

from transformers.trainer_callback import (
    ExportableState,
)

import re
from liger_kernel.chunked_loss.dpo_loss import LigerFusedLinearDPOLoss
from qwenvl.data.image_processing_qwen2_vl_fast import Qwen2VLImageProcessorFast

def _is_peft_model(model):
    # if is_peft_available():
    #     classes_to_check = (PeftModel,) if is_peft_available() else ()
    #     # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
    #     if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
    #         from peft import PeftMixedModel

    #         classes_to_check = (*classes_to_check, PeftMixedModel)
    #     return isinstance(model, classes_to_check)
    return False

# class DebugInferenceCallback(TrainerCallback):
#     """Callback to log debug inference to wandb"""
    
#     def __init__(self, model, tokenizer, data_args):
#         self.debug_log = os.environ.get("DEBUG_LOG_INPUTS_OUTPUTS", "0") == "1"
#         self.debug_frequency = int(os.environ.get("DEBUG_LOG_FREQUENCY", "50"))
#         self.debug_max_tokens = int(os.environ.get("DEBUG_MAX_NEW_TOKENS", "256"))
#         self.current_inputs = None
#         self.model = model
#         self.tokenizer = tokenizer
#         self.data_args = copy.deepcopy(data_args)
#         self.data_args.run_test = True
#         self.dataset = LazySupervisedDataset(tokenizer=self.tokenizer, data_args=self.data_args)
#         self.iterator = iter(DataLoader(
#             self.dataset,
#             batch_size=1,
#             shuffle=False,
#             num_workers=0,
#             collate_fn=lambda batch: batch[0],
#             in_order=False
#         ))

#     def on_step_end(self, args, state, control, **kwargs):    
#         logger.info(f"on_step_end called at step {state.global_step}")

#         if not self.debug_log or state.global_step % self.debug_frequency != 1:
#             logger.info(f"Skipping debug logging: debug_log={self.debug_log}, step={state.global_step}, frequency={self.debug_frequency}")
#             return
        
#         if self.current_inputs is None:
#             logger.info("No current_inputs available")
#             return

#         logger.info(f"Current inputs keys: {list(self.current_inputs.keys())}")
#         for k, v in self.current_inputs.items():
#             if v is None:
#                 logger.info(f"  {k}: None")
#             elif hasattr(v, 'shape'):
#                 logger.info(f"  {k}: shape {v.shape}, type {type(v)}")
#             else:
#                 logger.info(f"  {k}: type {type(v)}, value preview: {str(v)[:100]}")
                
#         logger.info("Starting debug inference...")
#         try:
#             log_data = {}
#             inputs = next(self.iterator)

#             res = {
#                 "video": inputs.pop("video", None),
#                 "image": inputs.pop("image", None),
#                 "prompt": inputs.pop("prompt", None),
#                 "ref": inputs.pop("ref", None),
#                 "audio": inputs.pop("audio", None),
#                 "use_audio": inputs.pop("use_audio", False),
#                 "should_use": inputs.pop("should_use", True),
#             }
            
#             inputs = {k: v.to(f"cuda:{torch.cuda.current_device()}") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
#             with torch.inference_mode():
#                 self.model.eval()

#                 print(self.model.training)

#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=1024,
#                     do_sample=True,
#                     top_p=0.9)

#                 self.model.train()
                
#             output_trimmed = outputs[0, len(inputs["input_ids"][0]):]
#             output_text = self.tokenizer.decode(output_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
#             res["pred"] = output_text # only one example for now

#             for video, pred, ref in zip(res["video"], [res["pred"]], [res["ref"]]):
#                 video.seek(0)
#                 log_data["debug_input_video"] = wandb.Video(video, format="mp4")
#                 log_data["debug_model_response"] = wandb.Html(f"<pre>{pred}</pre>")
#                 log_data["debug_reference"] = wandb.Html(f"<pre>{ref}</pre>")
                
#                 wandb.log(log_data, step=state.global_step, commit=True)

#         except Exception as e:
#             logger.error(f"Debug inference failed: {str(e)}")
#             logger.error(f"Full traceback: {traceback.format_exc()}")
#             logger.error(f"Available keys in current_inputs: {list(self.current_inputs.keys()) if self.current_inputs else 'None'}")

class QwenVLTrainer(Trainer):

    def __init__(
        self, 
        *args,
        **kwargs
    ):
        # Store data_args before calling super
        super().__init__(*args, **kwargs)
        self.dpo_loss_fct = LigerFusedLinearDPOLoss()
        
        # Add debug callback
        # self.debug_callback = DebugInferenceCallback(self.model, self.processing_class, self.data_args)
        # self.add_callback(self.debug_callback)

    
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]
                if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                    vision_tower_parameters = [
                        name for name, _ in opt_model.named_parameters() if "visual" in name
                    ]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n not in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and n in vision_tower_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n not in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def calc_dpo_loss(self, policy_input, policy_target, ref_input, ce_loss=None, beta=0.1):
        lm_head = self.model.lm_head.weight
        dpo_loss, (chosen_logp, reject_logp, chosen_logit, reject_logit, chosen_nll_loss, chosen_rewards, reject_rewards) = self.dpo_loss_fct(lm_head, policy_input, policy_target, ref_input=ref_input, ref_weight=lm_head)
        if ce_loss is not None:
            loss = dpo_loss + beta * ce_loss
        else:
            loss = dpo_loss
        print(f"RANK {dist.get_rank()} chosen: {chosen_rewards.item()}, reject: {reject_rewards.item()}")
        return (loss, dpo_loss, chosen_rewards, reject_rewards)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        train_type = inputs.get("train_type", "")
        if train_type == "sft":
            outputs = model(**inputs)
        elif train_type == "dpo":
            policy_input, policy_target = model(**inputs)
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(policy_input, policy_target, reference_input)
            
        elif train_type == "gdpo":
            policy_input, policy_target, ce_loss = model(**inputs)
            inputs["train_type"] = "dpo"
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                with torch.no_grad():
                    reference_input, reference_target = model(**inputs)
            outputs = self.calc_dpo_loss(policy_input, policy_target, reference_input, ce_loss=ce_loss)
        else:
            raise NotImplementedError

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss