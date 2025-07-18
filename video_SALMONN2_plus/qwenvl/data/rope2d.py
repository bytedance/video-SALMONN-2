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

import os
import copy
import json
import random
import logging
import re
import time
import math
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
# from decord import VideoReader
import transformers


def get_rope_index_25(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    audio_lengths: Optional[list] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embedding for text part.
        Examples:
            Temporal (Time): 3 patches, representing different segments of the video in time.
            Height: 2 patches, dividing each frame vertically.
            Width: 2 patches, dividing each frame horizontally.
            We also have some important parameters:
            fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
            tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
            temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
            interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [101, 102, 103, 104, 105]
            text height position_ids: [101, 102, 103 104, 105]
            text width position_ids: [101, 102, 103, 104, 105]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
            The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = 151655
    video_token_id = 151656
    audio_token_id = 151665
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (
        audio_lengths is not None and video_grid_thw is not None
    ):
        # 处理音视频交织输入的情况
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        video_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_videos = video_nums
            for _ in range(video_nums):
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0
                ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                video_pos = torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                # llm_pos_ids_list.append(video_pos)

                audio_len = audio_lengths[video_index]

                time_index_audio = torch.arange(audio_len, device=input_ids.device)
                w_index_audio = time_index_audio
                h_index_audio = torch.zeros_like(time_index_audio)
                audio_pos = torch.stack([time_index_audio, h_index_audio, w_index_audio]) + st_idx + text_len
                # llm_pos_ids_list.append(audio_pos)
                audio_visual_pos = torch.zeros_like(torch.cat((video_pos, audio_pos), dim=1))
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w + audio_len
                audio_visual_pos[:, input_ids[ed:st] == audio_token_id] = audio_pos
                audio_visual_pos[:, input_ids[ed:st] == video_token_id] = video_pos
                llm_pos_ids_list.append(audio_visual_pos)
                video_index += 1
                remain_videos -= 1

            if st < len(input_tokens):
                if len(llm_pos_ids_list) > 0:
                    # 取最后一个音频块的time维度（第0维）的最大值
                    last_time_max = llm_pos_ids_list[-1][0].max()
                    st_idx = last_time_max + 1
                else:
                    st_idx = 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    elif input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image

                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    if second_per_grid_ts is not None:
                        second_per_grid_t = second_per_grid_ts[video_index]
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                time_tensor = expanded_range * second_per_grid_t * 2

                time_tensor_long = time_tensor.long()
                t_index = time_tensor_long.flatten()

                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    elif input_ids is not None and audio_lengths is not None:
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        audio_index = 0
        attention_mask = attention_mask.to(total_input_ids.device)
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            audio_nums = 0
            # 定位音频起始位置（与视觉使用相同的起始token）
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            audio_nums = (vision_tokens == audio_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_audios = audio_nums
            for _ in range(audio_nums):
                # 查找当前音频结束位置
                if audio_token_id in input_tokens and remain_audios > 0:
                    ed_audio = input_tokens.index(audio_token_id, st)
                else:
                    ed_audio = len(input_tokens) + 1
                ed = ed_audio
                
                # 直接使用audio_lengths作为时间维度token数（用户说明是token个数）
                llm_grid_t = audio_lengths[audio_index]  # 时间维度token数（已直接是token个数）
                llm_grid_h = 1  # 音频无高度维度，固定为1
                llm_grid_w = 1  # 音频无宽度维度，固定为1
                
                text_len = ed - st
                st_idx = (llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0)
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
                time_index = torch.arange(llm_grid_t, device=input_ids.device)
                h_index = time_index  
                w_index = torch.zeros_like(time_index)  
                audio_pos = torch.stack([time_index, h_index, w_index]) + st_idx + text_len
                llm_pos_ids_list.append(audio_pos)
                
                st = ed + llm_grid_t 
                audio_index += 1
                remain_audios -= 1
            
            # 处理剩余文本部分（关键修复部分）
            if st < len(input_tokens):
                if len(llm_pos_ids_list) > 0:
                    # 取最后一个音频块的time维度（第0维）的最大值
                    last_time_max = llm_pos_ids_list[-1][0].max()
                    st_idx = last_time_max + 1
                else:
                    st_idx = 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )
            
            # 组合所有位置ID
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
        
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas

# original visual only rope index
# def get_rope_index_25(
#     spatial_merge_size: Optional[int] = 2,
#     input_ids: Optional[torch.LongTensor] = None,
#     image_grid_thw: Optional[torch.LongTensor] = None,
#     video_grid_thw: Optional[torch.LongTensor] = None,
#     second_per_grid_ts: Optional[torch.Tensor] = None,
#     attention_mask: Optional[torch.Tensor] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     """
#     Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

#     Explanation:
#         Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

#         For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
#         Examples:
#             input_ids: [T T T T T], here T is for text.
#             temporal position_ids: [0, 1, 2, 3, 4]
#             height position_ids: [0, 1, 2, 3, 4]
#             width position_ids: [0, 1, 2, 3, 4]

#         For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
#         and 1D rotary position embedding for text part.
#         Examples:
#             Temporal (Time): 3 patches, representing different segments of the video in time.
#             Height: 2 patches, dividing each frame vertically.
#             Width: 2 patches, dividing each frame horizontally.
#             We also have some important parameters:
#             fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
#             tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
#             temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
#             interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
#             input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
#             vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
#             vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
#             vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
#             text temporal position_ids: [101, 102, 103, 104, 105]
#             text height position_ids: [101, 102, 103, 104, 105]
#             text width position_ids: [101, 102, 103, 104, 105]
#             Here we calculate the text start position_ids as the max vision position_ids plus 1.

#     Args:
#         input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
#             Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
#             it.
#         image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
#             The temporal, height and width of feature shape of each image in LLM.
#         video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
#             The temporal, height and width of feature shape of each video in LLM.
#         second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
#             The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
#         attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.

#     Returns:
#         position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
#         mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
#     """
#     image_token_id = 151655
#     video_token_id = 151656
#     vision_start_token_id = 151652
#     mrope_position_deltas = []
#     if input_ids is not None and (
#         image_grid_thw is not None or video_grid_thw is not None
#     ):
#         total_input_ids = input_ids
#         if attention_mask is None:
#             attention_mask = torch.ones_like(total_input_ids)
#         position_ids = torch.ones(
#             3,
#             input_ids.shape[0],
#             input_ids.shape[1],
#             dtype=input_ids.dtype,
#             device=input_ids.device,
#         )
#         image_index, video_index = 0, 0
#         attention_mask = attention_mask.to(total_input_ids.device)
#         for i, input_ids in enumerate(total_input_ids):
#             input_ids = input_ids[attention_mask[i] == 1]
#             image_nums, video_nums = 0, 0
#             vision_start_indices = torch.argwhere(
#                 input_ids == vision_start_token_id
#             ).squeeze(1)
#             vision_tokens = input_ids[vision_start_indices + 1]
#             image_nums = (vision_tokens == image_token_id).sum()
#             video_nums = (vision_tokens == video_token_id).sum()
#             input_tokens = input_ids.tolist()
#             llm_pos_ids_list: list = []
#             st = 0
#             remain_images, remain_videos = image_nums, video_nums
#             for _ in range(image_nums + video_nums):
#                 if image_token_id in input_tokens and remain_images > 0:
#                     ed_image = input_tokens.index(image_token_id, st)
#                 else:
#                     ed_image = len(input_tokens) + 1
#                 if video_token_id in input_tokens and remain_videos > 0:
#                     ed_video = input_tokens.index(video_token_id, st)
#                 else:
#                     ed_video = len(input_tokens) + 1
#                 if ed_image < ed_video:
#                     t, h, w = (
#                         image_grid_thw[image_index][0],
#                         image_grid_thw[image_index][1],
#                         image_grid_thw[image_index][2],
#                     )
#                     second_per_grid_t = 0
#                     image_index += 1
#                     remain_images -= 1
#                     ed = ed_image

#                 else:
#                     t, h, w = (
#                         video_grid_thw[video_index][0],
#                         video_grid_thw[video_index][1],
#                         video_grid_thw[video_index][2],
#                     )
#                     if second_per_grid_ts is not None:
#                         second_per_grid_t = second_per_grid_ts[video_index]
#                     else:
#                         second_per_grid_t = 1.0
#                     video_index += 1
#                     remain_videos -= 1
#                     ed = ed_video
#                 llm_grid_t, llm_grid_h, llm_grid_w = (
#                     t.item(),
#                     h.item() // spatial_merge_size,
#                     w.item() // spatial_merge_size,
#                 )
#                 text_len = ed - st

#                 st_idx = (
#                     llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
#                 )
#                 llm_pos_ids_list.append(
#                     torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
#                 )

#                 range_tensor = torch.arange(llm_grid_t).view(-1, 1)
#                 expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

#                 time_tensor = expanded_range * second_per_grid_t * 2

#                 time_tensor_long = time_tensor.long()
#                 t_index = time_tensor_long.flatten()

#                 h_index = (
#                     torch.arange(llm_grid_h)
#                     .view(1, -1, 1)
#                     .expand(llm_grid_t, -1, llm_grid_w)
#                     .flatten()
#                 )
#                 w_index = (
#                     torch.arange(llm_grid_w)
#                     .view(1, 1, -1)
#                     .expand(llm_grid_t, llm_grid_h, -1)
#                     .flatten()
#                 )
#                 llm_pos_ids_list.append(
#                     torch.stack([t_index, h_index, w_index]) + text_len + st_idx
#                 )
#                 st = ed + llm_grid_t * llm_grid_h * llm_grid_w

#             if st < len(input_tokens):
#                 st_idx = (
#                     llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
#                 )
#                 text_len = len(input_tokens) - st
#                 llm_pos_ids_list.append(
#                     torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
#                 )

#             llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
#             position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
#                 position_ids.device
#             )
#             mrope_position_deltas.append(
#                 llm_positions.max() + 1 - len(total_input_ids[i])
#             )
#         mrope_position_deltas = torch.tensor(
#             mrope_position_deltas, device=input_ids.device
#         ).unsqueeze(1)
#         return position_ids, mrope_position_deltas
#     else:
#         if attention_mask is not None:
#             position_ids = attention_mask.long().cumsum(-1) - 1
#             position_ids.masked_fill_(attention_mask == 0, 1)
#             position_ids = (
#                 position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
#             )
#             max_position_ids = position_ids.max(0, keepdim=False)[0].max(
#                 -1, keepdim=True
#             )[0]
#             mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
#         else:
#             position_ids = (
#                 torch.arange(input_ids.shape[1], device=input_ids.device)
#                 .view(1, 1, -1)
#                 .expand(3, input_ids.shape[0], -1)
#             )
#             mrope_position_deltas = torch.zeros(
#                 [input_ids.shape[0], 1],
#                 device=input_ids.device,
#                 dtype=input_ids.dtype,
#             )

#         return position_ids, mrope_position_deltas


def get_rope_index_2(
    spatial_merge_size: Optional[int] = 2,
    input_ids: Optional[torch.LongTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

    Explanation:
        Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

        For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
        Examples:
            input_ids: [T T T T T], here T is for text.
            temporal position_ids: [0, 1, 2, 3, 4]
            height position_ids: [0, 1, 2, 3, 4]
            width position_ids: [0, 1, 2, 3, 4]

        For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
        and 1D rotary position embeddin for text part.
        Examples:
            Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
            input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
            vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
            vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
            vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
            text temporal position_ids: [3, 4, 5, 6, 7]
            text height position_ids: [3, 4, 5, 6, 7]
            text width position_ids: [3, 4, 5, 6, 7]
            Here we calculate the text start position_ids as the max vision position_ids plus 1.

    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
        image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
            The temporal, height and width of feature shape of each image in LLM.
        video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
            The temporal, height and width of feature shape of each video in LLM.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

    Returns:
        position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
        mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
    """
    image_token_id = 151655
    video_token_id = 151656
    vision_start_token_id = 151652
    mrope_position_deltas = []
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        total_input_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(total_input_ids)
        position_ids = torch.ones(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        image_index, video_index = 0, 0
        for i, input_ids in enumerate(total_input_ids):
            input_ids = input_ids[attention_mask[i] == 1]
            image_nums, video_nums = 0, 0
            vision_start_indices = torch.argwhere(
                input_ids == vision_start_token_id
            ).squeeze(1)
            vision_tokens = input_ids[vision_start_indices + 1]
            image_nums = (vision_tokens == image_token_id).sum()
            video_nums = (vision_tokens == video_token_id).sum()
            input_tokens = input_ids.tolist()
            llm_pos_ids_list: list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1
                if ed_image < ed_video:
                    t, h, w = (
                        image_grid_thw[image_index][0],
                        image_grid_thw[image_index][1],
                        image_grid_thw[image_index][2],
                    )
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = (
                        video_grid_thw[video_index][0],
                        video_grid_thw[video_index][1],
                        video_grid_thw[video_index][2],
                    )
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t, llm_grid_h, llm_grid_w = (
                    t.item(),
                    h.item() // spatial_merge_size,
                    w.item() // spatial_merge_size,
                )
                text_len = ed - st

                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

                t_index = (
                    torch.arange(llm_grid_t)
                    .view(-1, 1)
                    .expand(-1, llm_grid_h * llm_grid_w)
                    .flatten()
                )
                h_index = (
                    torch.arange(llm_grid_h)
                    .view(1, -1, 1)
                    .expand(llm_grid_t, -1, llm_grid_w)
                    .flatten()
                )
                w_index = (
                    torch.arange(llm_grid_w)
                    .view(1, 1, -1)
                    .expand(llm_grid_t, llm_grid_h, -1)
                    .flatten()
                )
                llm_pos_ids_list.append(
                    torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                )
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w

            if st < len(input_tokens):
                st_idx = (
                    llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                )
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                position_ids.device
            )
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(total_input_ids[i])
            )
        mrope_position_deltas = torch.tensor(
            mrope_position_deltas, device=input_ids.device
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = (
                position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
            )
            max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                -1, keepdim=True
            )[0]
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            mrope_position_deltas = torch.zeros(
                [input_ids.shape[0], 1],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )

        return position_ids, mrope_position_deltas
