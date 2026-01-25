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

import math
from typing import List, Optional, Union

import numpy as np
import torch

from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import AudioInput, ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput


class VideoSALMONN2PlusProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
            "return_mm_token_type_ids": False,
        },
    }


def _split_into_groups(counts, groups, second_per_grid_ts=None):
    result = []
    if second_per_grid_ts is None:
        for count, g in zip(counts, groups):
            g = int(g)
            base = count // g
            remainder = count % g
            if remainder == 0:
                group_list = [base] * g
            else:
                group_list = [base] * g
                step = g / remainder
                for i in range(1, remainder + 1):
                    position = i * step
                    index = math.floor(position) - 1
                    if index >= g:
                        index = g - 1
                    group_list[index] += 1
            result.append(group_list)
    else:
        for count, g, second in zip(counts, groups, second_per_grid_ts):
            g = int(g)
            frame_idx = (torch.arange(g) * second * 2).long()
            per_grid_t = torch.diff(frame_idx)
            group_list = per_grid_t.tolist()
            group_list.append(count - sum(group_list))
            result.append(group_list)
    return result


class VideoSALMONN2PlusProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer", "feature_extractor"]
    image_processor_class = "AutoImageProcessor"
    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        feature_extractor=None,
        chat_template=None,
        **kwargs,
    ):
        self.image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        self.video_token = getattr(tokenizer, "video_token", "<|video_pad|>")
        self.audio_token = getattr(tokenizer, "audio_token", "<|audio_pad|>")
        self.vision_start_token = getattr(tokenizer, "vision_start_token", "<|vision_start|>")
        self.vision_end_token = getattr(tokenizer, "vision_end_token", "<|vision_end|>")

        self.image_token_id = (
            tokenizer.image_token_id
            if getattr(tokenizer, "image_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if getattr(tokenizer, "video_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )
        self.audio_token_id = (
            tokenizer.audio_token_id
            if getattr(tokenizer, "audio_token_id", None)
            else tokenizer.convert_tokens_to_ids(self.audio_token)
        )

        super().__init__(image_processor, tokenizer, feature_extractor, chat_template=chat_template)

        self.auto_map = {
            "AutoProcessor": "qwenvl.model.processing_video_salmonn2_plus.VideoSALMONN2PlusProcessor"
        }

    def _get_num_visual_tokens(self, grid_thw, merge_size):
        if isinstance(grid_thw, torch.Tensor):
            num_patches = int(torch.prod(grid_thw).item())
        else:
            num_patches = int(np.prod(grid_thw))
        return num_patches // (merge_size**2)

    def _prepare_audio_features(self, audio: AudioInput):
        if audio is None:
            return None, None

        if isinstance(audio, (list, tuple)):
            audio_list = list(audio)
        else:
            audio_list = [audio]

        audio_inputs = []
        audio_lengths = []
        sampling_rate = getattr(self.feature_extractor, "sampling_rate", 16000)
        chunk_length = getattr(self.feature_extractor, "chunk_length", 30)
        segment_samples = int(sampling_rate * chunk_length)

        for audio_item in audio_list:
            if isinstance(audio_item, str):
                raise ValueError(
                    "Audio file paths are not supported in this processor. "
                    "Please provide raw audio arrays."
                )
            audio_array = np.array(audio_item, dtype=np.float32)
            if audio_array.ndim > 1:
                audio_array = np.mean(audio_array, axis=0)
            if audio_array.shape[0] < segment_samples:
                pad = segment_samples - audio_array.shape[0]
                audio_array = np.pad(audio_array, (0, pad), mode="constant", constant_values=0)

            segments = [
                audio_array[i : i + segment_samples]
                for i in range(0, audio_array.shape[0], segment_samples)
            ]
            segment_features = []
            for segment in segments:
                features = self.feature_extractor(
                    segment,
                    sampling_rate=sampling_rate,
                    padding="max_length",
                    return_attention_mask=False,
                    return_tensors="pt",
                )["input_features"].squeeze(0)
                segment_features.append(features)
            audio_inputs.append(torch.stack(segment_features, dim=0))
            audio_lengths.append(math.ceil(audio_array.shape[0] / segment_samples) * 60)

        audio_feature = torch.cat(audio_inputs, dim=0) if audio_inputs else None
        return audio_feature, audio_lengths

    def _expand_images(self, text, image_grid_thw, merge_size):
        if image_grid_thw is None:
            return text
        use_image_placeholders = any("<image>" in t for t in text)
        use_image_token_placeholders = False
        if not use_image_placeholders:
            total_image_tokens = sum(t.count(self.image_token) for t in text)
            use_image_token_placeholders = total_image_tokens == len(image_grid_thw)
        image_index = 0
        for i in range(len(text)):
            if "<image>" in text[i]:
                parts = text[i].split("<image>")
            elif use_image_token_placeholders and self.image_token in text[i]:
                parts = text[i].split(self.image_token)
            else:
                continue
            new_parts = []
            for part in parts[:-1]:
                new_parts.append(part)
                num_image_tokens = self._get_num_visual_tokens(image_grid_thw[image_index], merge_size)
                replacement = (
                    f"{self.vision_start_token}"
                    + (self.image_token * num_image_tokens)
                    + f"{self.vision_end_token}"
                )
                new_parts.append(replacement)
                image_index += 1
            new_parts.append(parts[-1])
            text[i] = "".join(new_parts)
        return text

    def _expand_audios(self, text, audio_lengths):
        if audio_lengths is None:
            return text
        use_audio_placeholders = any("<audio>" in t for t in text)
        use_audio_token_placeholders = False
        if not use_audio_placeholders:
            total_audio_tokens = sum(t.count(self.audio_token) for t in text)
            use_audio_token_placeholders = total_audio_tokens == len(audio_lengths)
        audio_index = 0
        for i in range(len(text)):
            if "<audio>" in text[i]:
                parts = text[i].split("<audio>")
            elif use_audio_token_placeholders and self.audio_token in text[i]:
                parts = text[i].split(self.audio_token)
            else:
                continue
            new_parts = []
            for part in parts[:-1]:
                new_parts.append(part)
                if audio_index >= len(audio_lengths):
                    raise ValueError("Not enough audio inputs for <audio> placeholders.")
                replacement = (
                    f"{self.vision_start_token}"
                    + (self.audio_token * audio_lengths[audio_index])
                    + f"{self.vision_end_token}"
                )
                new_parts.append(replacement)
                audio_index += 1
            new_parts.append(parts[-1])
            text[i] = "".join(new_parts)
        return text

    def _expand_videos(
        self,
        text,
        video_grid_thw,
        merge_size,
        audio_lengths=None,
        second_per_grid_ts=None,
        audio_for_video=False,
    ):
        if video_grid_thw is None:
            return text

        use_video_placeholders = any("<video>" in t for t in text)
        use_video_token_placeholders = False
        if not use_video_placeholders:
            total_video_tokens = sum(t.count(self.video_token) for t in text)
            use_video_token_placeholders = total_video_tokens == len(video_grid_thw)

        if audio_for_video and audio_lengths is not None:
            groups = [int(grid[0]) for grid in video_grid_thw]
            if second_per_grid_ts is not None:
                second_per_grid_ts = [ts[0] if isinstance(ts, (list, tuple)) else ts for ts in second_per_grid_ts]
            per_timestep_audio_len = _split_into_groups(audio_lengths, groups, second_per_grid_ts)
        else:
            per_timestep_audio_len = None

        video_index = 0
        for i in range(len(text)):
            if "<video>" in text[i]:
                parts = text[i].split("<video>")
            elif use_video_token_placeholders and self.video_token in text[i]:
                parts = text[i].split(self.video_token)
            else:
                continue
            new_parts = []
            for part in parts[:-1]:
                new_parts.append(part)
                if video_index >= len(video_grid_thw):
                    raise ValueError("Not enough video inputs for <video> placeholders.")
                if audio_for_video and audio_lengths is not None:
                    t = int(video_grid_thw[video_index][0])
                    h = int(video_grid_thw[video_index][1])
                    w = int(video_grid_thw[video_index][2])
                    tokens_per_frame = (h * w) // (merge_size**2)
                    replacement = f"{self.vision_start_token}"
                    for step in range(t):
                        replacement += (self.video_token * tokens_per_frame)
                        replacement += (self.audio_token * per_timestep_audio_len[video_index][step])
                    replacement += f"{self.vision_end_token}"
                else:
                    num_video_tokens = self._get_num_visual_tokens(video_grid_thw[video_index], merge_size)
                    replacement = (
                        f"{self.vision_start_token}"
                        + (self.video_token * num_video_tokens)
                        + f"{self.vision_end_token}"
                    )
                new_parts.append(replacement)
                video_index += 1
            new_parts.append(parts[-1])
            text[i] = "".join(new_parts)
        return text

    def _check_special_mm_tokens(self, text, text_inputs, modalities):
        for modality in modalities:
            token_str = getattr(self, f"{modality}_token")
            token_id = getattr(self, f"{modality}_token_id")
            ids_count = [list(ids).count(token_id) for ids in text_inputs["input_ids"]]
            text_count = [sample.count(token_str) for sample in text]
            if ids_count != text_count:
                raise ValueError(
                    f"Mismatch in `{modality}` token count between text and `input_ids`. "
                    f"Got ids={ids_count} and text={text_count}. "
                    "Likely due to `truncation='max_length'`. Please disable truncation or increase `max_length`."
                )

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: Optional[VideoInput] = None,
        audio: Optional[AudioInput] = None,
        second_per_grid_ts: Optional[List] = None,
        **kwargs: Unpack[VideoSALMONN2PlusProcessorKwargs],
    ) -> BatchFeature:
        if images is None and text is None and videos is None and audio is None:
            raise ValueError(
                "You need to provide at least one input to call VideoSALMONN2PlusProcessor."
            )

        output_kwargs = self._merge_kwargs(
            VideoSALMONN2PlusProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = {}
        video_inputs = {}
        image_grid_thw = None
        video_grid_thw = None
        if images is not None:
            image_inputs = self.image_processor(images=images, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs.get("image_grid_thw")

        if videos is not None:
            video_inputs = self.image_processor(videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = video_inputs.get("video_grid_thw")

        audio_feature, audio_lengths = self._prepare_audio_features(audio)

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text = text.copy()

            merge_size = getattr(self.image_processor, "merge_size", 2)
            text = self._expand_images(text, image_grid_thw, merge_size)
            audio_for_video = audio_lengths is not None and videos is not None and all("<audio>" not in t for t in text)
            if audio_for_video and video_grid_thw is not None and len(audio_lengths) != len(video_grid_thw):
                raise ValueError("Number of audio inputs must match number of videos when audio is interleaved.")

            text = self._expand_videos(
                text,
                video_grid_thw,
                merge_size,
                audio_lengths=audio_lengths,
                second_per_grid_ts=second_per_grid_ts,
                audio_for_video=audio_for_video,
            )
            text = self._expand_audios(text, audio_lengths)

            return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
            return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", False)
            text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)

            modalities = []
            if images is not None:
                modalities.append("image")
            if videos is not None:
                modalities.append("video")
            if audio_lengths is not None:
                modalities.append("audio")
            if modalities:
                self._check_special_mm_tokens(text, text_inputs, modalities=modalities)

            if return_mm_token_type_ids:
                array_ids = np.array(text_inputs["input_ids"])
                mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
                mm_token_type_ids[array_ids == self.image_token_id] = 1
                mm_token_type_ids[array_ids == self.video_token_id] = 2
                mm_token_type_ids[array_ids == self.audio_token_id] = 3
                text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        else:
            text_inputs = {}
            return_tensors = output_kwargs["common_kwargs"].get("return_tensors", None)

        audio_inputs = {}
        if audio_feature is not None:
            audio_inputs["audio_feature"] = audio_feature
            audio_inputs["audio_lengths"] = audio_lengths

        if second_per_grid_ts is not None:
            audio_inputs["second_per_grid_ts"] = second_per_grid_ts

        batch = BatchFeature(
            data={**text_inputs, **image_inputs, **video_inputs, **audio_inputs},
            tensor_type=return_tensors,
        )
        for key in ("audio_lengths", "second_per_grid_ts"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].tolist()
        return batch

    def post_process_image_text_to_text(
        self, generated_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False, **kwargs
    ):
        return self.tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )


__all__ = ["VideoSALMONN2PlusProcessor", "VideoSALMONN2PlusProcessorKwargs"]
