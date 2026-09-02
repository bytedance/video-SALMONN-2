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
# Adapted from Huggingface transformers and vLLM.
"""Inference-only Qwen3VL model compatible with HuggingFace weights."""

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import lru_cache, partial
from itertools import islice
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLImageProcessorFast
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    smart_resize as image_smart_resize,
)
from transformers.models.qwen3_vl import Qwen3VLProcessor, Qwen3VLVideoProcessor
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig as HFQwen3VLConfig,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    smart_resize as video_smart_resize,
)
from transformers.video_utils import VideoMetadata

from qwenvl.model.configuration_qwen3_vl import (
    Qwen3VLConfig as LocalQwen3VLConfig,
    Qwen3VLVisionConfig,
)
from qwenvl.model.configuration_qwen3_vl import WhisperConfig
from qwenvl.model.vllm_whisper import WhisperEncoder

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions, VideoDummyOptions
from vllm.distributed import get_pp_group, parallel_state
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import _ACTIVATION_REGISTRY
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.conv import Conv3dLayer
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.evs import (
    compute_mrope_for_media,
    compute_retained_tokens_count,
    compute_retention_mask,
    recompute_mrope_positions,
)
from vllm.multimodal.inputs import (
    MultiModalDataDict,
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    MultiModalKwargsItems,
    PlaceholderRange,
    VideoItem,
    ModalityData
)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems, MultiModalDataParser, ModalityDataItems, DictEmbeddingItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
    PromptInsertion,
    PromptIndexTargets,
)
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.sequence import IntermediateTensors
from vllm.tokenizers.protocol import TokenizerLike
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.utils.collection_utils import is_list_of
from vllm.utils.math_utils import round_up

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsMultiModalPruning,
    SupportsPP,
    _require_is_multimodal,
)
from vllm.model_executor.models.qwen2_5_vl import (
    Qwen2_5_VisionAttention,
    Qwen2_5_VLImageEmbeddingInputs,
    Qwen2_5_VLImageInputs,
    Qwen2_5_VLImagePixelInputs,
    Qwen2_5_VLVideoEmbeddingInputs,
    Qwen2_5_VLVideoInputs,
    Qwen2_5_VLVideoPixelInputs,
)
from vllm.model_executor.models.qwen2_vl import (
    Qwen2VLMultiModalDataParser,
    Qwen2VLProcessingInfo,
    _create_qwen2vl_field_factory,
)
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    _merge_multimodal_embeddings,
    maybe_prefix,
)
from vllm.model_executor.models.vision import (
    get_vit_attn_backend,
    is_vit_use_data_parallel,
    run_dp_sharded_mrope_vision_model,
)

logger = init_logger(__name__)
from vllm.utils.tensor_schema import TensorSchema, TensorShape
from typing import Annotated, Any, Literal, TypeAlias

# We use 2048 dummy video frames that would generate vision embeddings
# of the maximum size.
DUMMY_VIDEO_NUM_FRAMES = 2048

# ---------------------------------------------------------------------------
# Custom video loader backend + audio decoding for `vllm serve`.
#
# vLLM's default (OpenCV) video loader only decodes image frames and discards
# the raw bytes before they reach `_call_hf_processor`, so we cannot recover
# the audio track there. To feed raw video files through the OpenAI-compatible
# server while reusing the *training-time* sampling / audio logic from
# `qwenvl/data/data_qwen.py`, we register our own `VideoLoader` backend whose
# `load_bytes(data: bytes, ...)` receives the original bytes. It:
#   1. samples frames with the same interval / min-max frame logic used by
#      `LazySupervisedDataset.video_torchcodec`, at native resolution (the HF
#      video processor performs the smart-resize, matching training), and
#   2. probes for an audio stream and, if present, decodes the waveform and
#      stashes it (plus the sampled frame indices / fps) inside `metadata` so
#      that `Qwen3VLMultiModalProcessor._call_hf_processor` can build the
#      interleaved audio pad tokens exactly like training.
#
# Enable it with:
#   VLLM_VIDEO_LOADER_BACKEND=qwen3vl vllm serve <model> ...
# and tune sampling via
#   --media-io-kwargs '{"video": {"base_interval": 0.1, "video_max_frames": 128}}'
# ---------------------------------------------------------------------------

# Defaults mirror scripts/inference.py / DataArguments so that serving matches
# the offline preprocessing path.
QWEN3VL_VIDEO_DEFAULTS = {
    "base_interval": 0.1,
    "video_min_frames": 32,
    "video_max_frames": 900,
    "video_min_frame_pixels": 4 * 28 * 28,
    "video_max_frame_pixels": 128000,
    "use_audio": True,
    "audio_sampling_rate": 16000,
}


def _qwen3vl_sample_frame_indices(
    total_frame_num: int,
    avg_fps: float,
    *,
    base_interval: float,
    video_min_frames: int,
    video_max_frames: int,
) -> np.ndarray:
    """Replicate the frame-index sampling from data_qwen.video_torchcodec."""
    video_length = total_frame_num / avg_fps if avg_fps > 0 else 0
    num_frames_to_sample = round(video_length / base_interval) if base_interval else 0
    target_frames = min(
        max(num_frames_to_sample, video_min_frames), video_max_frames
    )
    target_frames = max(target_frames, 1)
    frame_idx = np.linspace(0, max(total_frame_num - 1, 0), target_frames, dtype=int)
    frame_idx = np.unique(frame_idx)
    return frame_idx


def _qwen3vl_decode_audio(data: bytes, sampling_rate: int):
    """Decode the audio track from raw video bytes.

    Returns the mono waveform as a float32 numpy array, or ``None`` when the
    container has no decodable audio stream.
    """
    from io import BytesIO

    from torchcodec.decoders import AudioDecoder

    try:
        decoder = AudioDecoder(
            BytesIO(data),
            sample_rate=sampling_rate,
            num_channels=1,
        )
        audio = decoder.get_all_samples()
    except Exception as exc:  # no audio stream / undecodable
        logger.info("Qwen3VL video backend: no usable audio stream (%s)", exc)
        return None
    return audio.data.numpy().squeeze(0)


def _register_qwen3vl_video_backend() -> None:
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader

    if "qwen3vl" in getattr(VIDEO_LOADER_REGISTRY, "_registry", {}):
        return

    @VIDEO_LOADER_REGISTRY.register("qwen3vl")
    class Qwen3VLVideoBackend(VideoLoader):
        @classmethod
        def load_bytes(
            cls,
            data: bytes,
            num_frames: int = -1,
            **kwargs,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            from io import BytesIO

            base_interval = kwargs.get(
                "base_interval", QWEN3VL_VIDEO_DEFAULTS["base_interval"]
            )
            video_min_frames = kwargs.get(
                "video_min_frames", QWEN3VL_VIDEO_DEFAULTS["video_min_frames"]
            )
            video_max_frames = kwargs.get(
                "video_max_frames", QWEN3VL_VIDEO_DEFAULTS["video_max_frames"]
            )
            use_audio = kwargs.get("use_audio", QWEN3VL_VIDEO_DEFAULTS["use_audio"])
            audio_sampling_rate = kwargs.get(
                "audio_sampling_rate", QWEN3VL_VIDEO_DEFAULTS["audio_sampling_rate"]
            )

            # --- decode frames (torchcodec, decord fallback) ---
            frame_idx = None
            avg_fps = None
            frames = None
            try:
                from torchcodec.decoders import VideoDecoder

                decoder = VideoDecoder(BytesIO(data), device="cpu")
                total_frame_num = decoder.metadata.num_frames
                avg_fps = decoder.metadata.average_fps
                frame_idx = _qwen3vl_sample_frame_indices(
                    total_frame_num,
                    avg_fps,
                    base_interval=base_interval,
                    video_min_frames=video_min_frames,
                    video_max_frames=video_max_frames,
                )
                frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
                # (T, C, H, W) -> (T, H, W, C) uint8 for the HF video processor.
                frames = frame_batch.data.cpu().numpy().transpose(0, 2, 3, 1)
            except Exception as exc:
                logger.warning(
                    "Qwen3VL video backend: torchcodec failed (%s), "
                    "falling back to decord.",
                    exc,
                )
                from decord import VideoReader, cpu

                vr = VideoReader(BytesIO(data), ctx=cpu(0), num_threads=1)
                total_frame_num = len(vr)
                avg_fps = vr.get_avg_fps()
                frame_idx = _qwen3vl_sample_frame_indices(
                    total_frame_num,
                    avg_fps,
                    base_interval=base_interval,
                    video_min_frames=video_min_frames,
                    video_max_frames=video_max_frames,
                )
                frames = vr.get_batch(frame_idx.tolist()).asnumpy()

            frame_idx_list = frame_idx.tolist()

            metadata: dict[str, Any] = {
                "total_num_frames": int(total_frame_num),
                "fps": float(avg_fps) if avg_fps else 0.0,
                "duration": (
                    float(total_frame_num / avg_fps) if avg_fps else 0.0
                ),
                "video_backend": "qwen3vl",
                "frames_indices": frame_idx_list,
                # We already sampled frames; tell the HF processor NOT to
                # resample (matches data_qwen: do_sample_frames=False).
                "do_sample_frames": False,
            }

            # --- decode audio track (optional) ---
            if use_audio:
                waveform = _qwen3vl_decode_audio(data, audio_sampling_rate)
                if waveform is not None and waveform.shape[0] > 0:
                    metadata["audio_waveform"] = waveform
                    metadata["audio_sampling_rate"] = audio_sampling_rate

            return frames, metadata


_register_qwen3vl_video_backend()


class Qwen3_VLVideoPixelInputs(TensorSchema):
    """
    Dimensions:
        - np: Number of patches
        - nv: Number of videos
        - ctps: Number of channels * temporal_patch_size * patch_size *
          patch_size

    Historical context:
        - pixel_values_videos shape: (num_patches, num_channels *
          temporal_patch_size * patch_size * patch_size)
        - video_grid_thw shape: (num_videos, 3) in (grid_t, grid_h, grid_w)
          format
        - second_per_grid_ts: The video time interval (in seconds) for each
          grid along the temporal dimension in the 3D position IDs. Returned
          when `videos` is not `None`.
        - timestamps: List of timestamp values (in seconds) for each frame
          after merging. Length equals the temporal dimension after merging.
    """

    type: Literal["pixel_values_videos"]

    pixel_values_videos: Annotated[
        torch.Tensor,
        TensorShape("np", "ctps"),
    ]

    video_grid_thw: Annotated[
        torch.Tensor,
        TensorShape("nv", 3),
    ]

    second_per_grid_ts: Annotated[
        torch.Tensor | None,
        TensorShape("nv"),
    ]

    timestamps: list[list[float]] | None = None

    audio_feature: Annotated[
        torch.Tensor | None,
        TensorShape("na", "mel", "dim"),
    ]

    per_prompt_size: list[int] | None = None

    per_prompt_audio: list[int] | None = None


def _create_qwen3vl_field_factory(
    spatial_merge_size: int,
) -> Callable[
    [Mapping[str, torch.Tensor]],
    Mapping[str, MultiModalFieldConfig],
]:
    def _qwen3vl_field_config(hf_inputs: Mapping[str, torch.Tensor]):
        if "timestamps" not in hf_inputs:
            image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
            image_pixel_grid_sizes = image_grid_thw.prod(-1)
            image_embed_grid_sizes = (
                image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
            )
            return dict(
                pixel_values=MultiModalFieldConfig.flat_from_sizes(
                    "image", image_pixel_grid_sizes
                ),
                image_embeds=MultiModalFieldConfig.flat_from_sizes(
                    "image", image_embed_grid_sizes
                ),
                image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
                pixel_values_videos=MultiModalFieldConfig.batched(
                    "video"
                ),
                video_embeds=MultiModalFieldConfig.batched(
                    "video"
                ),
                video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
                timestamps=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
                video_expanded_ids=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
                audio_feature=MultiModalFieldConfig.batched(
                    "audio"
                ),
            )
        else:
            image_grid_thw = hf_inputs.get("image_grid_thw", torch.empty((0, 3)))
            image_pixel_grid_sizes = image_grid_thw.prod(-1)
            image_embed_grid_sizes = (
                image_pixel_grid_sizes // spatial_merge_size // spatial_merge_size
            )

            video_grid_thw = hf_inputs.get("video_grid_thw", torch.empty((0, 3)))
            video_grid_sizes = video_grid_thw.prod(-1)
            video_embed_grid_sizes = (
                video_grid_sizes // spatial_merge_size // spatial_merge_size
            )
            return dict(
                pixel_values=MultiModalFieldConfig.flat_from_sizes(
                    "image", image_pixel_grid_sizes
                ),
                image_embeds=MultiModalFieldConfig.flat_from_sizes(
                    "image", image_embed_grid_sizes
                ),
                image_grid_thw=MultiModalFieldConfig.batched("image", keep_on_cpu=True),
                pixel_values_videos=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_grid_sizes
                ),
                video_embeds=MultiModalFieldConfig.flat_from_sizes(
                    "video", video_embed_grid_sizes
                ),
                video_grid_thw=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
                timestamps=MultiModalFieldConfig.batched("video", keep_on_cpu=True),
                video_expanded_ids=MultiModalFieldConfig.batched(
                    "video", keep_on_cpu=True
                ),
                audio_feature=MultiModalFieldConfig.batched(
                    "audio"
                ),
            )

    return _qwen3vl_field_config


def _create_qwen3vl_audio_field_factory() -> Callable[
    [Mapping[str, torch.Tensor]],
    Mapping[str, MultiModalFieldConfig],
]:
    def _qwen3vl_audio_field_config(hf_inputs: Mapping[str, torch.Tensor]):
        return dict(
            audio_feature=MultiModalFieldConfig.batched("audio"),
        )

    return _qwen3vl_audio_field_config


class Qwen3_VisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        hidden_size: int = 1152,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.hidden_size = hidden_size

        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = Conv3dLayer(
            in_channels,
            hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L, C = x.shape
        x = x.view(L, -1, self.temporal_patch_size, self.patch_size, self.patch_size)
        x = self.proj(x).view(L, self.hidden_size)
        return x


class Qwen3_VisionMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        bias: bool = False,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.linear_fc1 = ColumnParallelLinear(
            in_features,
            hidden_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.linear_fc2 = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )
        self.act_fn = act_fn

    def forward(self, x: torch.Tensor):
        mlp_output = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return mlp_output


class Qwen3_VisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        norm_layer: Callable[[int], nn.Module] | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = Qwen2_5_VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.mlp = Qwen3_VisionMLP(
            dim,
            mlp_hidden_dim,
            act_fn=act_fn,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        max_seqlen: torch.Tensor,  # Only used for Flash Attention
        sequence_lengths: torch.Tensor,  # Only used for FlashInfer CuDNN backend
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
            max_seqlen=max_seqlen,
            sequence_lengths=sequence_lengths,
        )

        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3_VisionPatchMerger(nn.Module):
    def __init__(
        self,
        d_model: int,
        context_dim: int,
        norm_layer: Callable[[int], nn.Module] | None = None,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        use_data_parallel = is_vit_use_data_parallel()
        self.hidden_size = context_dim * (spatial_merge_size**2)

        self.use_postshuffle_norm = use_postshuffle_norm
        if self.use_postshuffle_norm:
            context_dim = self.hidden_size

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm = norm_layer(context_dim)
        self.linear_fc1 = ColumnParallelLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc1",
            disable_tp=use_data_parallel,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = RowParallelLinear(
            self.hidden_size,
            d_model,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.linear_fc2",
            disable_tp=use_data_parallel,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)

        x_parallel, _ = self.linear_fc1(x)
        x_parallel = self.act_fn(x_parallel)
        out, _ = self.linear_fc2(x_parallel)
        return out


class Qwen3_VisionTransformer(nn.Module):
    def __init__(
        self,
        vision_config: Qwen3VLVisionConfig,
        norm_eps: float = 1e-6,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads
        self.num_position_embeddings = vision_config.num_position_embeddings
        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.temporal_patch_size = vision_config.temporal_patch_size
        self.deepstack_visual_indexes = (
            vision_config.deepstack_visual_indexes
            if hasattr(vision_config, "deepstack_visual_indexes")
            else []
        )
        self.num_grid_per_side = int(self.num_position_embeddings**0.5)

        use_data_parallel = is_vit_use_data_parallel()
        self.tp_size = (
            1
            if use_data_parallel
            else parallel_state.get_tensor_model_parallel_world_size()
        )

        # NOTE: This is used for creating empty tensor for all_gather for
        # DP ViT. Here out_hidden_size is enlarged due to deepstack
        self.out_hidden_size = vision_config.out_hidden_size * (
            1 + len(self.deepstack_visual_indexes)
        )

        self.patch_embed = Qwen3_VisionPatchEmbed(
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            in_channels=vision_config.in_channels,
            hidden_size=self.hidden_size,
        )

        self.pos_embed = nn.Embedding(self.num_position_embeddings, self.hidden_size)

        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            max_position=8192,
            is_neox_style=True,
            rope_parameters={"partial_rotary_factor": 0.5},
        )

        self.merger = Qwen3_VisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=self.hidden_size,
            norm_layer=norm_layer,
            spatial_merge_size=self.spatial_merge_size,
            quant_config=quant_config,
            prefix=f"{prefix}.merger",
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3_VisionPatchMerger(
                    d_model=vision_config.out_hidden_size,
                    context_dim=self.hidden_size,
                    spatial_merge_size=self.spatial_merge_size,
                    use_postshuffle_norm=True,
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.deepstack_merger_list.{layer_idx}",
                )
                for layer_idx in range(len(self.deepstack_visual_indexes))
            ]
        )

        self.attn_backend = get_vit_attn_backend(
            head_size=head_dim,
            dtype=torch.get_default_dtype(),
        )

        self.blocks = nn.ModuleList(
            [
                Qwen3_VisionBlock(
                    dim=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_hidden_dim=vision_config.intermediate_size,
                    act_fn=_ACTIVATION_REGISTRY[vision_config.hidden_act],
                    norm_layer=norm_layer,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(vision_config.depth)
            ]
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    def rot_pos_emb(self, grid_thw: list[list[int]]):
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        pos_ids = [
            self.rot_pos_ids(h, w, self.spatial_merge_size)
            if t == 1
            else self.rot_pos_ids(h, w, self.spatial_merge_size).repeat(t, 1)
            for t, h, w in grid_thw
        ]
        pos_ids = torch.cat(pos_ids, dim=0).to(self.device, non_blocking=True)

        # Use pre-computed cos_sin_cache from RotaryEmbedding
        cos, sin = self.rotary_pos_emb.get_cos_sin(max_grid_size)

        cos_combined = cos[pos_ids].flatten(1)
        sin_combined = sin[pos_ids].flatten(1)

        return cos_combined, sin_combined

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        num_grid_per_side = self.num_grid_per_side
        m_size = self.spatial_merge_size
        hidden_dim = self.pos_embed.embedding_dim

        outputs = []
        for t, h, w in grid_thw:
            h_idxs = torch.linspace(
                0, num_grid_per_side - 1, h, dtype=torch.float32, device=self.device
            )
            w_idxs = torch.linspace(
                0, num_grid_per_side - 1, w, dtype=torch.float32, device=self.device
            )

            h_floor = h_idxs.to(torch.long)
            w_floor = w_idxs.to(torch.long)
            h_ceil = torch.clamp(h_floor + 1, max=num_grid_per_side - 1)
            w_ceil = torch.clamp(w_floor + 1, max=num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            # Create meshgrid view for all h, w vars
            dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
            h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
            h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

            # original computation of weights
            # w00 = (1 - dh_grid) * (1 - dw_grid)
            # w01 = (1 - dh_grid) * dw_grid
            # w10 = dh_grid * (1 - dw_grid)
            # w11 = dh_grid * dw_grid
            # we reuse w11 here to avoid duplicate
            # dh_grid * dw_grid computation
            w11 = dh_grid * dw_grid
            w10 = dh_grid - w11
            w01 = dw_grid - w11
            w00 = 1 - dh_grid - w01

            h_grid = torch.stack([h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid])
            w_grid = torch.stack([w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid])
            h_grid_idx = h_grid * num_grid_per_side

            indices = (h_grid_idx + w_grid).reshape(4, -1)
            weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
            weights = weights.to(dtype=self.dtype)

            embeds = self.pos_embed(indices)
            embeds *= weights
            combined = embeds.sum(dim=0)

            combined = combined.reshape(
                h // m_size, m_size, w // m_size, m_size, hidden_dim
            )
            combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, hidden_dim)
            repeated = combined.expand(t, -1, -1).reshape(-1, hidden_dim)
            outputs.append(repeated)

        return torch.cat(outputs, dim=0)

    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor | list[list[int]],
    ) -> torch.Tensor:
        hidden_states = x.to(device=self.device, dtype=self.dtype, non_blocking=True)
        hidden_states = self.patch_embed(hidden_states)

        if isinstance(grid_thw, list):
            grid_thw_list = grid_thw
            grid_thw = np.array(grid_thw, dtype=np.int32)
        else:
            grid_thw_list = grid_thw.tolist()
            grid_thw = grid_thw.numpy()

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw_list)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb_cos, rotary_pos_emb_sin = self.rot_pos_emb(grid_thw_list)

        cu_seqlens = np.repeat(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            axis=0, dtype=np.int32
        )
        cu_seqlens = np.concatenate([np.zeros(1, dtype=np.int32), cu_seqlens])
        sequence_lengths = MMEncoderAttention.maybe_compute_sequence_lengths(
            self.attn_backend, cu_seqlens
        )
        if sequence_lengths is not None:
            sequence_lengths = torch.from_numpy(sequence_lengths).to(
                self.device, non_blocking=True
            )
        max_seqlen = torch.tensor(
            MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens),
            dtype=torch.int32,
            device=self.device,
        )
        cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
            self.attn_backend,
            cu_seqlens,
            self.hidden_size,
            self.tp_size,
        )
        cu_seqlens = torch.from_numpy(cu_seqlens).to(self.device, non_blocking=True)
        hidden_states = hidden_states.unsqueeze(1)

        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                max_seqlen=max_seqlen,
                sequence_lengths=sequence_lengths,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[deepstack_merger_idx](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        hidden_states = torch.cat(
            [hidden_states] + deepstack_feature_lists, dim=1
        )  # [seq_len, hidden_size * (1 + depth_of_deepstack)]
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("attn.qkv.", "attn.q.", "q"),
            ("attn.qkv.", "attn.k.", "k"),
            ("attn.qkv.", "attn.v.", "v"),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

class Qwen3VLMultiModalDataParser(Qwen2VLMultiModalDataParser):
    def __init__(self, spatial_merge_size: int, *args, **kwargs):
        super().__init__(spatial_merge_size, *args, **kwargs)

    def _parse_video_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[VideoItem],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="video",
                required_fields={"video_grid_thw"},
                fields_factory=_create_qwen3vl_field_factory(self._spatial_merge_size),
            )

        return super()._parse_video_data(data)

    def _parse_audio_data(
        self,
        data: dict[str, torch.Tensor] | ModalityData[Any],
    ) -> ModalityDataItems[Any, Any] | None:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_feature"},
                fields_factory=_create_qwen3vl_audio_field_factory(),
            )

        return super()._parse_audio_data(data)


class Qwen3VLProcessingInfo(Qwen2VLProcessingInfo):
    def get_supported_mm_limits(self):
        return {"audio": None, "image": None, "video": None}

    def get_hf_config(self):
        try:
            hf_config = self.ctx.get_hf_config(LocalQwen3VLConfig)
        except TypeError:
            hf_config = self.ctx.get_hf_config(HFQwen3VLConfig)
        if getattr(hf_config, "audio_config", None) is None:
            hf_config.audio_config = WhisperConfig()
            hf_config.audio_config.qformer_config.out_hidden_size = (
                hf_config.text_config.hidden_size
            )
        if getattr(hf_config, "audio_token_id", None) is None:
            hf_config.audio_token_id = 151669
        return hf_config

    def get_hf_processor(self, **kwargs: object) -> Qwen3VLProcessor:
        return self.ctx.get_hf_processor(
            Qwen3VLProcessor,
            use_fast=kwargs.pop("use_fast", True),
            **kwargs,
        )

    def get_image_processor(self, **kwargs: object) -> Qwen2VLImageProcessorFast:
        return self.get_hf_processor(**kwargs).image_processor

    def get_video_processor(self, **kwargs: object) -> Qwen3VLVideoProcessor:
        return self.get_hf_processor(**kwargs).video_processor

    def get_audio_feature_extractor(self):
        """Whisper mel feature extractor used to build ``audio_feature``.

        Mirrors ``scripts/inference.py`` / ``DataArguments`` defaults so that
        serving matches the offline preprocessing path. Cached on the instance
        to avoid rebuilding per request.
        """
        extractor = getattr(self, "_audio_feature_extractor", None)
        if extractor is None:
            from transformers import WhisperFeatureExtractor

            extractor = WhisperFeatureExtractor(
                feature_size=128,
                sampling_rate=QWEN3VL_VIDEO_DEFAULTS["audio_sampling_rate"],
                hop_length=160,
                chunk_length=30,
            )
            self._audio_feature_extractor = extractor
        return extractor

    def get_data_parser(self):
        return Qwen3VLMultiModalDataParser(
            self.get_hf_config().vision_config.spatial_merge_size,
            video_needs_metadata=True,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def _get_vision_info(
        self,
        *,
        image_width: int,
        image_height: int,
        num_frames: int = 2,
        do_resize: bool = True,
        image_processor: Qwen2VLImageProcessorFast | Qwen3VLVideoProcessor,
        mm_kwargs: Mapping[str, object],
    ) -> tuple[ImageSize, int]:
        is_video = isinstance(image_processor, Qwen3VLVideoProcessor)

        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config
        patch_size = vision_config.patch_size
        merge_size = vision_config.spatial_merge_size
        temporal_patch_size = vision_config.temporal_patch_size

        mm_kwargs = self.ctx.get_merged_mm_kwargs(mm_kwargs)
        size = image_processor.size
        if override_size := mm_kwargs.get("size"):
            size = size | override_size
        if (override_min_pixels := mm_kwargs.get("min_pixels")) is not None:
            size = size | {"shortest_edge": override_min_pixels}
        if (override_max_pixels := mm_kwargs.get("max_pixels")) is not None:
            size = size | {"longest_edge": override_max_pixels}

        if do_resize:
            if is_video:
                smart_resize = video_smart_resize
                extra_kwargs = {
                    "num_frames": num_frames,
                    "temporal_factor": temporal_patch_size,
                }
            else:
                smart_resize = image_smart_resize
                extra_kwargs = {}

            resized_height, resized_width = smart_resize(
                height=image_height,
                width=image_width,
                factor=patch_size * merge_size,
                min_pixels=size["shortest_edge"],
                max_pixels=size["longest_edge"],
                **extra_kwargs,
            )
            preprocessed_size = ImageSize(width=resized_width, height=resized_height)
        else:
            preprocessed_size = ImageSize(width=image_width, height=image_height)

        padded_num_frames = round_up(num_frames, temporal_patch_size)

        grid_t = max(padded_num_frames // temporal_patch_size, 1)
        grid_h = preprocessed_size.height // patch_size
        grid_w = preprocessed_size.width // patch_size

        num_patches = grid_t * grid_h * grid_w
        num_vision_tokens = num_patches // (merge_size**2)

        return preprocessed_size, num_vision_tokens

    def _get_max_video_frames(self, max_tokens: int, start_num_frames: int = 2) -> int:
        return super()._get_max_video_frames(
            max_tokens, start_num_frames=start_num_frames
        )

    def get_num_frames_with_most_features(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        return super().get_num_frames_with_most_features(
            seq_len, mm_counts, max_frames_per_video=DUMMY_VIDEO_NUM_FRAMES
        )

    def get_max_video_tokens(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> int:
        video_processor = self.get_video_processor()

        mm_kwargs = self.ctx.get_merged_mm_kwargs({})
        video_size = mm_kwargs.get("size", video_processor.size)
        temporal_patch_size = mm_kwargs.get(
            "temporal_patch_size", video_processor.temporal_patch_size
        )

        # video_max_pixels contains the temporal compression factor,
        # so we divide by 2 to get the maximum number of image pixels.
        video_max_pixels = video_size["longest_edge"]
        target_width, target_height = self.get_image_size_with_most_features(
            max_pixels=video_max_pixels // temporal_patch_size
        )
        num_video_soft_tokens = self.get_num_video_tokens(
            image_width=target_width,
            image_height=target_height,
            num_frames=2,
            image_processor=video_processor,
            mm_kwargs={},
        )
        return num_video_soft_tokens

    def _calculate_timestamps(
        self, indices: list[int] | torch.Tensor, video_fps: float, merge_size: int
    ):
        if not isinstance(indices, list):
            indices = indices.tolist()
        if len(indices) % merge_size != 0:
            # don't update metadata's frames_indices directly
            indices = indices + [indices[-1]] * (merge_size - len(indices) % merge_size)
        timestamps = [idx / video_fps for idx in indices]
        timestamps = [
            (timestamps[i] + timestamps[i + merge_size - 1]) / 2
            for i in range(0, len(timestamps), merge_size)
        ]
        return timestamps

    def _get_video_second_idx(
        self,
        metadata: dict[str, Any],
        do_sample_frames: bool | None = None,
        sampled_fps: float | None = None,
    ) -> list[int]:
        video_processor = self.get_video_processor()
        merge_size = video_processor.merge_size
        indices = metadata["frames_indices"]

        # metadata["fps"] refers to the true fps of the input video.
        video_fps = metadata["fps"]
        if do_sample_frames is None:
            do_sample_frames = metadata.get("do_sample_frames", False)

        # If video frames are sampled in HF processor (instead of vLLM
        # video loader), we need to re-calculate the indices from original
        # metadata.
        if do_sample_frames:
            # here video_fps is the fps of the sampled video, and
            # metadata["fps"] refers to the fps of the original video.
            sampled_fps = sampled_fps if sampled_fps else video_processor.fps
            total_num_frames = metadata["total_num_frames"]
            num_frames = int(total_num_frames / metadata["fps"] * sampled_fps)
            num_frames = min(
                min(
                    max(num_frames, video_processor.min_frames),
                    video_processor.max_frames,
                ),
                total_num_frames,
            )
            indices = (
                np.linspace(0, total_num_frames - 1, num_frames)
                .round()
                .astype(int)
                .tolist()
            )
        timestamps = self._calculate_timestamps(indices, video_fps, merge_size)
        return timestamps


class Qwen3VLDummyInputsBuilder(BaseDummyInputsBuilder[Qwen3VLProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)

        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        video_token = "<|vision_start|><|video_pad|><|vision_end|>"

        # NOTE: audio is a derived, interleaved-in-video modality (route A):
        # audio pad tokens live INSIDE the video block and are emitted by the
        # video PromptReplacement, not as standalone tokens. So the dummy text
        # only contains the minimal video marker; the processor expands it into
        # the full per-frame interleaved (video + audio) sequence.
        return image_token * num_images + video_token * num_videos

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        num_videos = mm_counts.get("video", 0)
        num_audios = mm_counts.get("audio", 0)
        image_overrides = mm_options.get("image")
        video_overrides = mm_options.get("video")

        target_image_width, target_image_height = (
            self.info.get_image_size_with_most_features()
        )

        # treat videos as special images
        target_num_frames = 2
        if video_overrides:
            assert isinstance(video_overrides, VideoDummyOptions)
            num_frames_override = video_overrides.num_frames
            if num_frames_override:
                if num_frames_override > target_num_frames:
                    logger.warning(
                        "video.num_frames override (%d) exceeds model's "
                        "maximum number of frames (%d), will be ignored",
                        num_frames_override,
                        target_num_frames,
                    )
                if num_frames_override < 2:
                    logger.warning(
                        "video.num_frames override (%d) cannot be less "
                        "than 2, will be ignored",
                        num_frames_override,
                    )
                target_num_frames = min(target_num_frames, num_frames_override)
        target_num_frames = max(target_num_frames, 2)

        video_processor = self.info.get_video_processor()

        mm_kwargs = self.info.ctx.get_merged_mm_kwargs({})
        video_size = mm_kwargs.get("size", video_processor.size)
        temporal_patch_size = mm_kwargs.get(
            "temporal_patch_size", video_processor.temporal_patch_size
        )

        # video_max_pixels contains the temporal compression factor,
        # so we divide by 2 to get the maximum number of image pixels.
        video_max_pixels = video_size["longest_edge"]
        target_video_width, target_video_height = (
            self.info.get_image_size_with_most_features(
                max_pixels=video_max_pixels // temporal_patch_size
            )
        )
        target_video_size, _ = self.info._get_vision_info(
            image_width=target_video_width,
            image_height=target_video_height,
            num_frames=target_num_frames,
            image_processor=video_processor,
            mm_kwargs={},
        )
        # NOTE: we need to do this check here since Qwen3-VL resizes video
        # frames depending on how many frames there are.
        target_video_width, target_video_height = (
            target_video_size.width,
            target_video_size.height,
        )
        if video_overrides:
            assert isinstance(video_overrides, VideoDummyOptions)
            width_override = video_overrides.width
            if width_override:
                if width_override > target_video_width:
                    logger.warning(
                        "video.width override (%d) exceeds model's "
                        "maximum width (%d), will be ignored",
                        width_override,
                        target_video_width,
                    )
                target_video_width = min(target_video_width, width_override)
            height_override = video_overrides.height
            if height_override:
                if height_override > target_video_height:
                    logger.warning(
                        "video.height override (%d) exceeds model's "
                        "maximum height (%d), will be ignored",
                        height_override,
                        target_video_height,
                    )
                target_video_height = min(target_video_height, height_override)

        out: MultiModalDataDict = {
            "image": self._get_dummy_images(
                width=target_image_width,
                height=target_image_height,
                num_images=num_images,
                overrides=image_overrides,
            ),
            "video": self._get_dummy_videos(
                width=target_video_width,
                height=target_video_height,
                num_frames=target_num_frames,
                num_videos=num_videos,
            ),
        }

        if num_audios:
            hf_config = self.info.get_hf_config()
            # Audio is a derived, interleaved-in-video modality (route A): each
            # audio item is paired 1:1 with a video item and its pad tokens live
            # inside that video's vision block. Therefore we can only emit as
            # many dummy audios as there are dummy videos, and we pass the audio
            # as precomputed `audio_feature` embeddings (dict form) matching the
            # real inference contract in `_parse_and_validate_audio_input`.
            num_audios = min(num_audios, num_videos)
            if num_audios:
                num_mel_bins = hf_config.audio_config.num_mel_bins
                max_source_positions = hf_config.audio_config.max_source_positions
                expected_seq_length = max_source_positions * 2
                # One chunk per audio item -> `1 * 60` interleaved audio tokens,
                # distributed across the dummy video's frames by the video
                # PromptReplacement.
                num_chunks = 1
                out["audio"] = {
                    "audio_feature": torch.zeros(
                        (
                            num_audios,
                            num_chunks,
                            num_mel_bins,
                            expected_seq_length,
                        ),
                        dtype=torch.float32,
                    )
                }

        return out

    def _get_dummy_videos(
        self,
        *,
        width: int,
        height: int,
        num_frames: int,
        num_videos: int,
    ) -> list[VideoItem]:
        video = np.full((num_frames, width, height, 3), 255, dtype=np.uint8)
        video_items = []
        for i in range(num_videos):
            video_metadata = {
                "fps": 2.0,
                "duration": num_frames / 2.0,
                "total_num_frames": num_frames,
                "frames_indices": [i for i in range(num_frames)],
                "video_backend": "opencv",
                "do_sample_frames": False,
            }
            video_item = (video.copy(), video_metadata)
            video_items.append(video_item)
        return video_items


def _compute_audio_feature(
    waveform: np.ndarray,
    feature_extractor,
    sampling_rate: int,
) -> tuple[torch.Tensor, int]:
    """Compute Whisper mel features + token length from a mono waveform.

    Faithfully replicates ``LazySupervisedDataset.process_audio`` in
    ``qwenvl/data/data_qwen.py`` (the ``timestamps is None`` single-clip case):

    - pad clips shorter than 1s to exactly 1s,
    - split into 30s chunks,
    - run each chunk through the Whisper feature extractor,
    - stack the per-chunk mel spectrograms into ``(num_chunks, mel, T)``,
    - token length is ``ceil(len / (30 * sr)) * 60``.

    Returns ``(audio_feature, audio_length)``.
    """
    chunk_samples = 30 * sampling_rate
    if waveform.shape[0] < sampling_rate:
        pad = sampling_rate - waveform.shape[0]
        waveform = np.pad(waveform, (0, pad), mode="constant", constant_values=0)

    chunks = [
        waveform[k : k + chunk_samples]
        for k in range(0, len(waveform), chunk_samples)
    ]
    spectrogram_lst = [
        feature_extractor(
            chunk, sampling_rate=sampling_rate, return_tensors="pt"
        )["input_features"].squeeze()
        for chunk in chunks
    ]
    audio_feature = torch.stack(spectrogram_lst, dim=0)
    audio_length = math.ceil(len(waveform) / chunk_samples) * 60
    assert audio_length % 60 == 0, "Weird audio length"
    return audio_feature, audio_length


def _split_audio_into_frames(
    audio_total: int,
    num_frames: int,
    second_per_grid_t: float | None,
) -> list[int]:
    """Distribute ``audio_total`` audio tokens across ``num_frames`` frames.

    Faithfully replicates ``split_into_groups`` from
    ``qwenvl/data/data_qwen.py`` (single-video case):

    - When ``second_per_grid_t`` is provided, follow the
      ``second_per_grid_ts is not None`` branch (timestamp-derived split).
    - Otherwise, follow the even-distribution branch.

    Returns a list[int] of length ``num_frames`` that sums to ``audio_total``.
    """
    if second_per_grid_t is None:
        base = audio_total // num_frames
        remainder = audio_total % num_frames
        if remainder == 0:
            group_list = [base] * num_frames
        else:
            group_list = [base] * num_frames
            step = num_frames / remainder
            for i in range(1, remainder + 1):
                position = i * step
                index = math.floor(position) - 1
                if index >= num_frames:
                    index = num_frames - 1
                group_list[index] += 1
    else:
        frame_idx = (torch.arange(num_frames) * second_per_grid_t * 2).long()
        per_grid_t = torch.diff(frame_idx)
        group_list = per_grid_t.tolist()
        group_list.append(audio_total - sum(group_list))
        assert sum(group_list) == audio_total, (
            f"Strage count under count={audio_total}, g={num_frames}, "
            f"second={second_per_grid_t}"
        )
    return group_list


class Qwen3VLMultiModalProcessor(BaseMultiModalProcessor[Qwen3VLProcessingInfo]):
    def apply(self, *args, **kwargs):
        result = super().apply(*args, **kwargs)
        # In the OpenAI-server path, audio is derived from the video's own
        # audio stream inside `_call_hf_processor` (there is no standalone
        # top-level "audio" input item). vLLM therefore produces `audio`
        # entries in `mm_kwargs` / `mm_placeholders` but NO `audio` entry in
        # `mm_hashes` (hashes are computed from input items only). The V1
        # input processor indexes `mm_hashes[modality][idx]` for every
        # placeholder modality, so a missing `audio` hash raises
        # `KeyError: 'audio'`. Synthesize deterministic audio hashes derived
        # from the co-located video hashes to fill the gap.
        mm_kwargs = result.get("mm_kwargs")
        mm_hashes = result.get("mm_hashes")
        if mm_kwargs is not None and mm_hashes is not None:
            audio_items = mm_kwargs.get("audio")
            if audio_items and "audio" not in mm_hashes:
                video_hashes = mm_hashes.get("video", [])
                audio_hashes = []
                for idx in range(len(audio_items)):
                    base = (
                        video_hashes[idx]
                        if idx < len(video_hashes)
                        else f"qwen3vl-audio-{idx}"
                    )
                    audio_hashes.append(f"{base}-audio")
                mm_hashes["audio"] = audio_hashes
        return result

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        mm_data = dict(mm_data)
        processor = self.info.get_hf_processor(**mm_kwargs)

        # Separate video processing from image processing. Because the videos
        # are processed into several image patches
        if videos := mm_data.pop("videos", []):
            video_grid_thw_lst = []
            pixel_values_videos_lst = []
            timestamps_per_video = []
            audio_feature_lst = []
            video_expanded_ids_lst = []

            for item in videos:
                video_array, metadata = item

                # NOTE: @JJJYmmm new attr metadata.frames_indices indicates
                # the sampled frames indices of pre-sampled videos, which is
                # used to calculate the timestamps. Make sure that
                # do_sample_frames in mm_kwargs is false for presampled videos.

                # NOTE: a copy of is created to update do_sample_frames,
                # otherwise mm_hash for the object will be incorrect.
                video_mm_kwargs = dict(**mm_kwargs)
                if "do_sample_frames" not in video_mm_kwargs:
                    # qwen_vl_utils already has "do_sample_frames" in
                    # mm_kwargs, don't overwrite it.
                    video_mm_kwargs["do_sample_frames"] = metadata.get(
                        "do_sample_frames", False
                    )

                # The custom qwen3vl video backend (see
                # `_register_qwen3vl_video_backend`) stashes the decoded audio
                # waveform in the metadata dict. Pull it out here BEFORE the
                # VideoMetadata conversion (which would reject unknown keys).
                audio_waveform = metadata.get("audio_waveform")
                audio_sampling_rate = metadata.get(
                    "audio_sampling_rate",
                    QWEN3VL_VIDEO_DEFAULTS["audio_sampling_rate"],
                )

                metadata = VideoMetadata(
                    **{
                        k: metadata[k]
                        for k in metadata
                        if k
                        not in (
                            "do_sample_frames",
                            "audio_waveform",
                            "audio_sampling_rate",
                        )
                    }
                )

                # Compute timestamps here where we have access to metadata
                timestamps = self.info._get_video_second_idx(
                    metadata=metadata,
                    do_sample_frames=video_mm_kwargs["do_sample_frames"],
                    sampled_fps=video_mm_kwargs.get("fps"),
                )
                timestamps_per_video.append(timestamps)

                video_mm_data = dict()
                video_mm_data["videos"] = [[video_array]]
                video_mm_data["video_metadata"] = [[metadata]]

                video_outputs = super()._call_hf_processor(
                    prompt="<|vision_start|><|video_pad|><|vision_end|>",
                    mm_data=video_mm_data,
                    mm_kwargs=video_mm_kwargs,
                    tok_kwargs=tok_kwargs,
                )

                merge_size = processor.video_processor.merge_size
                # Get video grid info for EVS calculation.
                video_grid_thw = video_outputs["video_grid_thw"]
                num_frames = int(video_grid_thw[0, 0])
                tokens_per_frame_base = int(video_grid_thw[0, 1:].prod()) // (
                    merge_size**2
                )

                # Apply EVS if enabled.
                video_pruning_rate = self.info.ctx.get_mm_config().video_pruning_rate
                if video_pruning_rate is not None and video_pruning_rate > 0.0:
                    num_tokens = compute_retained_tokens_count(
                        tokens_per_frame=tokens_per_frame_base,
                        num_frames=num_frames,
                        q=video_pruning_rate,
                    )
                    # Here we just need placeholders that won't actually be replaced -
                    # we just need to make sure the total number of tokens is correct
                    # assign all tokens to the first frame.
                    tokens_per_frame = [num_tokens] + [0] * (num_frames - 1)
                    select_token_id = False
                else:
                    tokens_per_frame = [tokens_per_frame_base] * num_frames
                    select_token_id = True

                # Generate the video replacement with EVS-adjusted token counts
                tokenizer = self.info.get_tokenizer()
                hf_config = self.info.get_hf_config()

                # Build the interleaved audio pad layout when the video carries
                # a decoded audio track. This mirrors data_qwen.py: audio pad
                # tokens are distributed across the video's frames (grid
                # temporal dim) via `split_into_groups` and emitted INSIDE each
                # frame's vision block, right before vision_end.
                audio_per_frame = None
                audio_token_id_arg = None
                if audio_waveform is not None:
                    feature_extractor = self.info.get_audio_feature_extractor()
                    audio_feature, audio_length = _compute_audio_feature(
                        audio_waveform,
                        feature_extractor,
                        audio_sampling_rate,
                    )
                    # second_per_grid_ts = temporal_patch_size / sampled_fps,
                    # matching process_video_frames in data_qwen.py.
                    sampled_fps = (
                        len(metadata.frames_indices) / metadata.duration
                        if metadata.duration
                        else 0.0
                    )
                    temporal_patch_size = (
                        processor.video_processor.temporal_patch_size
                    )
                    second_per_grid_t = (
                        temporal_patch_size / sampled_fps if sampled_fps else None
                    )
                    audio_per_frame = _split_audio_into_frames(
                        audio_length,
                        num_frames,
                        second_per_grid_t,
                    )
                    audio_token_id_arg = hf_config.audio_token_id
                    audio_feature_lst.append(audio_feature)

                video_repl = Qwen3VLMultiModalProcessor.get_video_repl(
                    tokens_per_frame=tokens_per_frame,
                    timestamps=timestamps,
                    tokenizer=tokenizer,
                    vision_start_token_id=hf_config.vision_start_token_id,
                    vision_end_token_id=hf_config.vision_end_token_id,
                    video_token_id=hf_config.video_token_id,
                    select_token_id=select_token_id,
                    audio_per_frame=audio_per_frame,
                    audio_token_id=audio_token_id_arg,
                )

                input_ids = video_outputs.pop("input_ids")
                if audio_per_frame is not None:
                    # With interleaved audio we must use our own layout (the HF
                    # processor does not emit audio pads).
                    video_placeholder = tokenizer.decode(
                        video_repl.full, skip_special_tokens=False
                    )
                else:
                    # Pure video: reuse the HF processor's expanded ids.
                    video_placeholder = processor.tokenizer.batch_decode(input_ids)[0]
                prompt = prompt.replace(
                    "<|vision_start|><|video_pad|><|vision_end|>",
                    video_placeholder,
                    1,
                )

                # Store the full per-frame expanded token layout (with
                # interleaved audio pads). It is emitted as an mm field so the
                # prompt-update expansion path (used when the OpenAI server
                # sends pre-tokenized prompts) can replace the single
                # `<|video_pad|>` target with this exact interleaved sequence
                # instead of a single collapsed vision block.
                video_expanded_ids_lst.append(
                    torch.tensor(list(video_repl.full), dtype=torch.long)
                )

                video_grid_thw_lst.append(video_outputs["video_grid_thw"])
                pixel_values_videos_lst.append(video_outputs["pixel_values_videos"])
            video_outputs = dict(
                pixel_values_videos=torch.cat(pixel_values_videos_lst),
                video_grid_thw=torch.cat(video_grid_thw_lst),
                timestamps=timestamps_per_video,
                video_expanded_ids=video_expanded_ids_lst,
            )
            if audio_feature_lst:
                video_outputs["audio_feature"] = audio_feature_lst
        else:
            video_outputs = dict()

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
            tok_kwargs=tok_kwargs,
        )
        combined_outputs = dict(
            processed_outputs,
            **video_outputs,
        )
        return BatchFeature(combined_outputs)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _create_qwen3vl_field_factory(
            self.info.get_hf_config().vision_config.spatial_merge_size
        )(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
        hf_config = self.info.get_hf_config()

        video_token_id = hf_config.video_token_id
        vision_start_token_id = hf_config.vision_start_token_id
        vision_end_token_id = hf_config.vision_end_token_id
        audio_token_id = hf_config.audio_token_id

        # vLLM eagerly resolves replacement callbacks for every multimodal item
        # before engine execution. The training prompt is already expanded, so
        # allocating one Python list per visual/audio token here only delays GPU
        # scheduling. Binding these constant replacements still records the
        # correct item counts; exact replacements are materialized lazily only
        # for paths that actually apply or inspect them.
        return [
            PromptReplacement(
                modality="image",
                target=hf_processor.image_token,
                replacement=[hf_processor.image_token_id],
            ),
            PromptReplacement(
                modality="audio",
                target="<|audio_pad|>",
                replacement=[audio_token_id],
            ),
            PromptReplacement(
                modality="video",
                target=hf_processor.video_token,
                replacement=PromptUpdateDetails.select_token_id(
                    [
                        vision_start_token_id,
                        video_token_id,
                        vision_end_token_id,
                    ],
                    video_token_id,
                ),
            ),
        ]

    def _materialize_prompt_updates(
        self,
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        *,
        include_audio: bool,
    ) -> MultiModalPromptUpdates:
        """Restore full replacement contents for non-preexpanded prompts."""
        hf_config = self.info.get_hf_config()
        merge_length = hf_config.vision_config.spatial_merge_size**2
        materialized = dict(mm_prompt_updates)

        if image_updates := mm_prompt_updates.get("image"):
            materialized["image"] = []
            for item_idx, item_updates in enumerate(image_updates):
                out_item = mm_kwargs["image"][item_idx]
                grid_thw = out_item["image_grid_thw"].data
                assert isinstance(grid_thw, torch.Tensor)
                replacement = (
                    [hf_config.image_token_id]
                    * (int(grid_thw.prod()) // merge_length)
                )
                materialized["image"].append(
                    [update.with_content(replacement) for update in item_updates]
                )

        if video_updates := mm_prompt_updates.get("video"):
            materialized["video"] = []
            for item_idx, item_updates in enumerate(video_updates):
                out_item = mm_kwargs["video"][item_idx]
                # Prefer the full per-frame expanded layout produced by
                # `_call_hf_processor` (one vision block per grid-t frame, with
                # interleaved audio pads). This is what the training/offline
                # path lays out, so the single `<|video_pad|>` target from the
                # OpenAI server's pre-tokenized prompt expands into exactly that
                # sequence.
                expanded = out_item.get("video_expanded_ids")
                if expanded is not None:
                    expanded_ids = expanded.data
                    if isinstance(expanded_ids, torch.Tensor):
                        expanded_ids = expanded_ids.tolist()
                    replacement = PromptUpdateDetails.select_token_id(
                        list(expanded_ids),
                        hf_config.video_token_id,
                    )
                else:
                    grid_thw = out_item["video_grid_thw"].data
                    assert isinstance(grid_thw, torch.Tensor)
                    num_tokens = int(grid_thw.prod()) // merge_length
                    replacement = PromptUpdateDetails.select_token_id(
                        [hf_config.vision_start_token_id]
                        + [hf_config.video_token_id] * num_tokens
                        + [hf_config.vision_end_token_id],
                        hf_config.video_token_id,
                    )
                materialized["video"].append(
                    [update.with_content(replacement) for update in item_updates]
                )

        if include_audio and (
            audio_updates := mm_prompt_updates.get("audio")
        ):
            materialized["audio"] = []
            for item_idx, item_updates in enumerate(audio_updates):
                out_item = mm_kwargs["audio"][item_idx]
                audio_feature = out_item["audio_feature"].data
                replacement = (
                    [hf_config.audio_token_id]
                    * (int(audio_feature.shape[0]) * 60)
                )
                materialized["audio"].append(
                    [update.with_content(replacement) for update in item_updates]
                )

        return materialized

    def _effective_mm_item_counts(
        self,
        mm_items: MultiModalDataItems,
        mm_kwargs: MultiModalKwargsItems,
    ) -> dict[str, int]:
        """Item counts including output-only modalities.

        When a video is submitted with an embedded audio track (the OpenAI /
        `vllm serve` path), audio is decoded inside the video backend and
        emitted as an ``audio_feature`` OUTPUT field. vLLM derives
        ``mm_item_counts`` from the INPUT ``mm_data`` only, so "audio" would be
        counted as 0 even though ``mm_kwargs["audio"]`` exists. We supplement
        the input counts with any modality that is present in ``mm_kwargs`` so
        that audio placeholders are located and validated correctly.
        """
        counts = dict(mm_items.get_all_counts())
        for modality in ("audio",):
            if modality not in counts:
                out_items = mm_kwargs.get(modality)
                if out_items:
                    counts[modality] = len(out_items)
        return counts

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        """Locate multimodal placeholders from the (already expanded) prompt.

        The inference prompt is fully pre-tokenized: every ``<|vision_start|>
        ... <|vision_end|>`` block already contains the exact interleaved
        image/video pad tokens (and, for use_audio_in_video, audio pad tokens
        interleaved inside the video block). We therefore do not re-derive the
        expanded sequence (which would require timestamps / fps); instead we
        scan the prompt directly and build one PlaceholderRange per multimodal
        item using pure token-id masks:

        - image: ``is_embed = (tokens == image_token_id)``
        - video: ``is_embed = (tokens == video_token_id)``
        - audio: shares the enclosing video block span, with
          ``is_embed = (tokens == audio_token_id)``

        Each feature is thus 1 modality / 1 mm_hash / 1 encoder output / 1
        is_embed mask, which is exactly what the V1 runner needs to slice
        embeddings per (possibly chunked) prefill window.
        """
        mm_item_counts = self._effective_mm_item_counts(mm_items, mm_kwargs)
        self._validate_mm_kwargs(mm_kwargs, mm_item_counts)

        if is_update_applied:
            # Our `_call_hf_processor` returns fully per-frame-expanded
            # `input_ids` (one vision block per grid-t frame, with interleaved
            # audio pads). vLLM adopts those ids as `prompt_ids` and marks
            # updates as applied. The single-block `_materialize_prompt_updates`
            # replacements cannot locate this interleaved layout, so we scan the
            # already-expanded prompt directly (same logic as the pre-tokenized
            # offline path).
            mm_placeholders = self._scan_expanded_prompt_placeholders(
                prompt_ids, mm_kwargs, mm_item_counts
            )
            self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
            return prompt_ids, mm_placeholders

        # ``is_update_applied`` is False both for the fully pre-tokenized
        # inference prompt (already expanded) and for the un-expanded
        # str/dummy prompt. Distinguish them by checking whether the vision pad
        # tokens are already fully present in the prompt.
        if self._prompt_is_expanded(prompt_ids, mm_kwargs, mm_item_counts):
            mm_placeholders = self._scan_expanded_prompt_placeholders(
                prompt_ids, mm_kwargs, mm_item_counts
            )
            self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
            return prompt_ids, mm_placeholders

        # Un-expanded prompt: this is the OpenAI-server / pre-tokenized-text
        # path where the prompt still holds a single `<|video_pad|>` per video.
        # We expand ONLY the video target; its materialized replacement is the
        # full per-frame interleaved layout (video + audio pads) produced by
        # `_call_hf_processor`. After expansion the prompt matches the offline
        # layout, so we scan it to build BOTH video and audio placeholders
        # (audio pads live inside the video block and have no standalone
        # target to expand).
        mm_prompt_updates = self._materialize_prompt_updates(
            mm_kwargs,
            mm_prompt_updates,
            include_audio=False,
        )
        prompt_ids, _ = self._apply_prompt_updates(
            prompt_ids,
            {k: v for k, v in mm_prompt_updates.items() if k != "audio"},
        )
        mm_placeholders = self._scan_expanded_prompt_placeholders(
            prompt_ids, mm_kwargs, mm_item_counts
        )
        self._validate_mm_placeholders(mm_placeholders, mm_item_counts)
        return prompt_ids, mm_placeholders

    def _prompt_is_expanded(
        self,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_item_counts: Mapping[str, int],
    ) -> bool:
        """Whether the vision pad tokens are already expanded in the prompt.

        There are only two kinds of prompt we ever see:

        - The fully pre-tokenized inference prompt, where every video/image
          block already contains its full run of pad tokens (>= 1 per item,
          and for any real video far more than 1).
        - The un-expanded ``str``/dummy prompt used for profiling, which
          contains exactly ONE pad token per item (a single ``<|video_pad|>``
          / ``<|image_pad|>`` that HF's prompt update later expands).

        The expansion path (`_apply_prompt_updates`) replaces EVERY existing
        pad token with a full ``num_tokens`` run, so running it on an
        already-expanded prompt causes a quadratic blow-up. We therefore only
        report "not expanded" when the prompt genuinely holds a single pad per
        item; any prompt with more pads than items is treated as expanded and
        located by scanning (a real count mismatch then fails cleanly in
        `_validate_mm_placeholders` instead of exploding).
        """
        hf_config = self.info.get_hf_config()
        prompt_tensor = torch.tensor(prompt_ids)

        def is_dummy(modality: str, token_id: int) -> bool:
            num_items = mm_item_counts.get(modality, 0)
            if num_items == 0:
                return False
            present = int((prompt_tensor == token_id).sum().item())
            # One pad per item => un-expanded dummy prompt.
            return present <= num_items

        if is_dummy("image", hf_config.image_token_id):
            return False
        if is_dummy("video", hf_config.video_token_id):
            return False
        return True

    def _scan_expanded_prompt_placeholders(
        self,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsItems,
        mm_item_counts: Mapping[str, int],
    ) -> Mapping[str, list[PlaceholderFeaturesInfo]]:
        """Build placeholders by scanning vision blocks in the expanded prompt.

        The training/inference prompt lays out a video as ONE
        ``<|vision_start|> ... <|vision_end|>`` block PER FRAME (see
        ``data_qwen.py``): each frame block holds ``frame_seq`` video pad tokens
        and, for use_audio_in_video, the frame's audio pad tokens, with a
        ``<t.t seconds>`` text prefix between frames. All frame blocks of one
        video are contiguous. We therefore merge the ``frames`` consecutive
        video blocks of each video item into a SINGLE PlaceholderRange spanning
        from the first frame's vision_start to the last frame's vision_end; the
        intervening timestamp text tokens get ``is_embed=False``.

        Images remain one block == one item.
        """
        hf_config = self.info.get_hf_config()
        vision_start_id = hf_config.vision_start_token_id
        vision_end_id = hf_config.vision_end_token_id
        image_token_id = hf_config.image_token_id
        video_token_id = hf_config.video_token_id
        audio_token_id = hf_config.audio_token_id
        merge_length = self.info.get_hf_processor().image_processor.merge_size ** 2

        # Per-video expected number of video pad tokens. A video is laid out as
        # ONE vision block per frame, so its frame blocks must be merged into a
        # single PlaceholderRange. We group consecutive video blocks by the
        # total video-pad count (grid.prod() // merge_length) rather than by a
        # frame count read from the grid's temporal dim, because the grid may be
        # represented either as [t, h, w] or as one [1, h, w] row per frame -
        # prod() is invariant to that, a bare [0] index is not.
        video_pad_targets: list[int] = []
        if mm_item_counts.get("video", 0) > 0:
            for item in mm_kwargs["video"]:
                grid = item["video_grid_thw"].data
                video_pad_targets.append(int(grid.prod()) // merge_length)

        # 1) Collect every vision block as (start_idx, end_idx_inclusive, kind).
        blocks: list[tuple[int, int, str]] = []
        n = len(prompt_ids)
        i = 0
        while i < n:
            if prompt_ids[i] != vision_start_id:
                i += 1
                continue
            j = i + 1
            while j < n and prompt_ids[j] != vision_end_id:
                j += 1
            if j >= n:
                break
            block = prompt_ids[i : j + 1]
            block_t = torch.tensor(block)
            if bool((block_t == image_token_id).any().item()):
                kind = "image"
            elif bool((block_t == video_token_id).any().item()):
                kind = "video"
            else:
                kind = "other"
            blocks.append((i, j, kind))
            i = j + 1

        placeholders: dict[str, list[PlaceholderFeaturesInfo]] = {
            "image": [],
            "video": [],
            "audio": [],
        }

        # 2) Walk blocks in order, grouping video frame-blocks per video item.
        video_idx = 0
        b = 0
        num_blocks = len(blocks)
        while b < num_blocks:
            start_idx, end_idx, kind = blocks[b]
            if kind == "image":
                span = prompt_ids[start_idx : end_idx + 1]
                span_t = torch.tensor(span)
                placeholders["image"].append(
                    PlaceholderFeaturesInfo(
                        modality="image",
                        item_idx=len(placeholders["image"]),
                        start_idx=start_idx,
                        tokens=span,
                        is_embed=span_t == image_token_id,
                    )
                )
                b += 1
            elif kind == "video":
                # Merge consecutive video blocks until we have accumulated this
                # video item's expected number of video pad tokens (one block
                # per frame). Counting pads (not blocks) is robust to how the
                # grid encodes the frame count.
                target_pads = (
                    video_pad_targets[video_idx]
                    if video_idx < len(video_pad_targets)
                    else 0
                )
                first_start = blocks[b][0]
                last_end = blocks[b][1]
                acc_pads = 0
                while b < num_blocks and blocks[b][2] == "video":
                    bs, be, _ = blocks[b]
                    block_ids = torch.tensor(prompt_ids[bs : be + 1])
                    acc_pads += int((block_ids == video_token_id).sum().item())
                    last_end = be
                    b += 1
                    if acc_pads >= target_pads:
                        break
                span = prompt_ids[first_start : last_end + 1]
                span_t = torch.tensor(span)
                placeholders["video"].append(
                    PlaceholderFeaturesInfo(
                        modality="video",
                        item_idx=len(placeholders["video"]),
                        start_idx=first_start,
                        tokens=span,
                        is_embed=span_t == video_token_id,
                    )
                )
                if bool((span_t == audio_token_id).any().item()):
                    placeholders["audio"].append(
                        PlaceholderFeaturesInfo(
                            modality="audio",
                            item_idx=len(placeholders["audio"]),
                            start_idx=first_start,
                            tokens=span,
                            is_embed=span_t == audio_token_id,
                        )
                    )
                video_idx += 1
            else:
                b += 1

        # Drop modalities not present in this request.
        return {
            modality: items
            for modality, items in placeholders.items()
            if mm_item_counts.get(modality, 0) > 0
        }

    @staticmethod
    def get_video_repl(
        *,
        tokens_per_frame: list[int],
        timestamps: list[float | int],
        tokenizer: TokenizerLike,
        vision_start_token_id: int,
        vision_end_token_id: int,
        video_token_id: int,
        select_token_id: bool = False,
        audio_per_frame: list[int] | None = None,
        audio_token_id: int | None = None,
    ) -> PromptUpdateDetails[list[int]]:
        """Build prompt replacement for a video in Qwen3VL format.

        The replacement structure for each frame is:
        timestamp_tokens + vision_start_token + video_tokens
        [+ audio_tokens] + vision_end_token

        Args:
            tokens_per_frame: Number of video tokens per frame (can vary per frame for
                EVS).
            timestamps: List of timestamps in seconds for each frame
            tokenizer: Tokenizer to encode timestamp strings
            vision_start_token_id: Token ID for vision start marker
            vision_end_token_id: Token ID for vision end marker
            video_token_id: Token ID for video content
            audio_per_frame: When provided, number of interleaved audio pad
                tokens to append per frame (inside the vision block, before
                vision_end). Must have the same length as ``timestamps``.
            audio_token_id: Token ID for interleaved audio pad content.

        Returns:
            PromptUpdateDetails with full token sequence
        """
        assert len(timestamps) == len(tokens_per_frame), (
            "timestamps and tokens_per_frame must have the same length"
        )
        if audio_per_frame is not None:
            assert audio_token_id is not None, (
                "audio_token_id must be provided when audio_per_frame is given"
            )
            assert len(audio_per_frame) == len(timestamps), (
                "audio_per_frame and timestamps must have the same length"
            )

        # Tokenize timestamp strings independently to avoid tokenizer merging
        # tokens across boundaries.
        # TODO: switch to `_seq2tokens` which has some caching.
        timestamp_token_ids = [
            tokenizer.encode(f"<{timestamp:.1f} seconds>", add_special_tokens=False)
            for timestamp in timestamps
        ]

        # Build the full token sequence
        all_token_ids = []
        for frame_idx, (frame_timestamp_ids, num_tokens) in enumerate(
            zip(timestamp_token_ids, tokens_per_frame)
        ):
            # Add timestamp tokens
            all_token_ids.extend(frame_timestamp_ids)

            # Add vision tokens: vision_start + video_tokens
            # [+ audio_tokens] + vision_end
            all_token_ids.append(vision_start_token_id)
            all_token_ids.extend([video_token_id] * num_tokens)
            if audio_per_frame is not None:
                all_token_ids.extend([audio_token_id] * audio_per_frame[frame_idx])
            all_token_ids.append(vision_end_token_id)

        if select_token_id:
            return PromptUpdateDetails.select_token_id(all_token_ids, video_token_id)

        # NOTE: we use `from_seq` instead of `select_token_id` because we want all
        # tokens in the placeholder to be initially marked as candidates. Then
        # in `get_input_embeddings``, we refine the mask to only replace
        # `video_token_id` / `image_token_id`` positions with video/image embeddings,
        # keeping text embeddings for timestamps and structural tokens.
        return PromptUpdateDetails.from_seq(all_token_ids)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        # the same shape as input_embeds
        "deepstack_input_embeds": 0,
    }
)
class Qwen3LLMModel(Qwen3Model):
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        # args for deepstack
        deepstack_input_embeds: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = []
        for layer_idx, layer in islice(
            enumerate(self.layers), self.start_layer, self.end_layer
        ):
            if layer_idx in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states + residual)

            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

            if deepstack_input_embeds is not None and layer_idx in range(
                0, len(deepstack_input_embeds)
            ):
                hidden_states = (
                    hidden_states
                    + deepstack_input_embeds[f"deepstack_input_embeds_{layer_idx}"]
                )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class Qwen3LLMForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super(Qwen3ForCausalLM, self).__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.quant_config = quant_config
        self.model = Qwen3LLMModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix="lm_head",
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor,
    info=Qwen3VLProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class Qwen3VLForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsLoRA,
    SupportsPP,
    SupportsMRoPE,
    SupportsEagle3,
    SupportsMultiModalPruning,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "qkv": ["qkv"],  # For vision tower's already-packed QKV
    }

    supports_encoder_tp_data = True

    # To ensure correct weight loading and mapping.
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.visual.": "visual.",
            "model.audio.": "audio.",
            "lm_head.": "language_model.lm_head.",
            "model.language_model.": "language_model.model.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        if modality.startswith("audio"):
            return "<|audio_pad|>"

        raise ValueError("Unsupported modality")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__()
        config = vllm_config.model_config.hf_config
        if getattr(config, "audio_config", None) is None:
            config.audio_config = WhisperConfig()
            config.audio_config.qformer_config.out_hidden_size = (
                config.text_config.hidden_size
            )
        if getattr(config, "audio_token_id", None) is None:
            config.audio_token_id = 151669
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.config = config
        self._tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.video_pruning_rate = multimodal_config.video_pruning_rate
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes)
            if self.use_deepstack
            else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )

            # register buffer for deepstack
            if self.use_deepstack:
                self.deepstack_input_embeds = [
                    torch.zeros(
                        vllm_config.scheduler_config.max_num_batched_tokens,
                        config.text_config.hidden_size,
                    )
                    for _ in range(self.deepstack_num_level)
                ]

        with self._mark_language_model(vllm_config):
            self.language_model = Qwen3LLMForCausalLM(
                vllm_config=vllm_config.with_hf_config(config.text_config),
                prefix=maybe_prefix(prefix, "language_model"),
            )

        with self._mark_tower_model(vllm_config, {"audio"}):
            audio_config = WhisperConfig(**config.audio_config)
            audio_config._attn_implementation="flash_attention_2"
            self.audio = WhisperEncoder(audio_config)

        if not get_pp_group().is_first_rank and hasattr(
            config.vision_config, "deepstack_visual_indexes"
        ):
            assert self.language_model.start_layer >= len(
                config.vision_config.deepstack_visual_indexes
            ), (
                "start_layer should be greater than or equal to "
                "len(deepstack_visual_indexes)"
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self._cached_audio_embeddings = None

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.language_model.model.aux_hidden_state_layers = layers

    def get_eagle3_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.language_model.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def _get_deepstack_input_embeds(
        self,
        num_tokens: int,
    ) -> IntermediateTensors | None:
        if not getattr(self, "deepstack_input_embeds", None):
            return None  # If vision tower is skipped

        # get deepstack_input_embeds from buffer, and clear the buffer
        return IntermediateTensors(
            {
                f"deepstack_input_embeds_{idx}": self.deepstack_input_embeds[idx][
                    :num_tokens
                ]
                for idx in range(self.deepstack_num_level)
            }
        )

    def _set_deepstack_input_embeds(self, deepstack_input_embeds: torch.Tensor) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return

        # set deepstack_input_embeds to buffer
        num_tokens = deepstack_input_embeds.size(1)
        if num_tokens > self.deepstack_input_embeds[0].size(0):
            self.deepstack_input_embeds = [
                torch.zeros(
                    num_tokens,
                    self.config.text_config.hidden_size,
                    device=self.deepstack_input_embeds[0].device,
                    dtype=self.deepstack_input_embeds[0].dtype,
                )
                for _ in range(self.deepstack_num_level)
            ]
        for idx in range(self.deepstack_num_level):
            self.deepstack_input_embeds[idx][:num_tokens].copy_(
                deepstack_input_embeds[idx]
            )

    def _clear_deepstack_input_embeds(self, num_tokens: int) -> None:
        if not getattr(self, "deepstack_input_embeds", None):
            return

        # clear deepstack_input_embeds in buffer
        if num_tokens > 0:
            for idx in range(self.deepstack_num_level):
                self.deepstack_input_embeds[idx][:num_tokens].zero_()

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> Qwen2_5_VLImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return Qwen2_5_VLImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )

        if image_embeds is not None:
            return Qwen2_5_VLImageEmbeddingInputs(
                type="image_embeds",
                image_embeds=image_embeds,
                image_grid_thw=image_grid_thw,
            )

    def _parse_and_validate_video_input(
        self, **kwargs: object
    ) -> Qwen3_VLVideoPixelInputs | None:
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        video_embeds = kwargs.pop("video_embeds", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        second_per_grid_ts = kwargs.pop("second_per_grid_ts", None)
        timestamps = kwargs.pop("timestamps", None)

        if pixel_values_videos is None and video_embeds is None:
            return None

        if not isinstance(pixel_values_videos, list) and pixel_values_videos.ndim == 3:
            per_prompt_size = [pixel_values_videos.shape[1] for _ in range(pixel_values_videos.shape[0])]
            pixel_values_videos = pixel_values_videos.flatten(0, 1)
        elif isinstance(pixel_values_videos, list):
            per_prompt_size = [v.shape[0] for v in pixel_values_videos]
            pixel_values_videos = torch.cat(pixel_values_videos, dim=0)
        else:
            per_prompt_size = [pixel_values_videos.shape[0]]

        if video_grid_thw.ndim == 3:
            video_grid_thw = video_grid_thw.flatten(0, 1)

        if pixel_values_videos is not None:
            return Qwen3_VLVideoPixelInputs(
                type="pixel_values_videos",
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                timestamps=timestamps,
                audio_feature=None,
                per_prompt_size=per_prompt_size,
                per_prompt_audio=None,
            )

        if video_embeds is not None:
            return Qwen2_5_VLVideoEmbeddingInputs(
                type="video_embeds",
                video_embeds=video_embeds,
                video_grid_thw=video_grid_thw,
                timestamps=timestamps,
            )

    def _parse_and_validate_audio_input(self, **kwargs: object) -> dict | None:
        audio_feature = kwargs.pop("audio_feature", None)
        if audio_feature is None:
            return None

        # Each audio item produces `chunks * 60` embedding tokens. Flatten the
        # per-item audio features into a single (num_chunks, mel, dim) tensor
        # while recording how many embedding tokens each item contributes so
        # that `_process_audio_input` can split them back into per-item tensors.
        if isinstance(audio_feature, list):
            per_prompt_audio = [v.shape[0] * 60 for v in audio_feature]
            audio_feature = torch.cat(audio_feature, dim=0)
        elif audio_feature.ndim == 4:
            per_prompt_audio = [
                audio_feature.shape[1] * 60 for _ in range(audio_feature.shape[0])
            ]
            audio_feature = audio_feature.flatten(0, 1)
        else:
            per_prompt_audio = [audio_feature.shape[0] * 60]

        return dict(
            audio_feature=audio_feature,
            per_prompt_audio=per_prompt_audio,
        )

    def _process_image_input(
        self, image_input: Qwen2_5_VLImageInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = image_input["image_grid_thw"]
        assert grid_thw.ndim == 2

        if image_input["type"] == "image_embeds":
            image_embeds = image_input["image_embeds"].type(self.visual.dtype)
        else:
            pixel_values = image_input["pixel_values"].type(self.visual.dtype)
            if self.use_data_parallel:
                return run_dp_sharded_mrope_vision_model(
                    self.visual, pixel_values, grid_thw.tolist(), rope_type="rope_3d"
                )
            else:
                image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Split concatenated embeddings for each image item.
        merge_size = self.visual.spatial_merge_size
        sizes = (grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return image_embeds.split(sizes)

    def _process_video_input(
        self, video_input: Qwen2_5_VLVideoInputs
    ) -> tuple[torch.Tensor, ...]:
        grid_thw = video_input["video_grid_thw"]
        per_prompt_size = video_input["per_prompt_size"]
        assert grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(
                self.visual.dtype
            )
            if self.use_data_parallel:
                grid_thw_list = grid_thw.tolist()
                video_embeds = torch.cat(
                    run_dp_sharded_mrope_vision_model(self.visual, pixel_values_videos, grid_thw_list, rope_type="rope_3d"), dim=0
                )
            else:
                video_embeds = self.visual(pixel_values_videos, grid_thw=grid_thw)

        # Split concatenated embeddings for each video item.
        merge_size = self.visual.spatial_merge_size
        sizes = [s // merge_size // merge_size for s in per_prompt_size]
        return video_embeds.split(sizes)

    def _process_audio_input(self, audio_input: dict) -> torch.Tensor:
        audio_feature = audio_input["audio_feature"]
        per_prompt_audio = audio_input["per_prompt_audio"]
        audio_feature = audio_feature.type(self.audio.dtype)

        # --- 修改部分开始 ---
        chunk_size = 20
        # 在第一维（dim=0）按长度 20 进行拆分
        chunks = torch.split(audio_feature, chunk_size, dim=0)
        
        processed_chunks = []
        for chunk in chunks:
            # 逐段输入到 self.audio
            processed_chunk = self.audio(chunk)
            processed_chunks.append(processed_chunk)
        
        # 将处理后的结果在第一维重新拼接
        audio_feature = torch.cat(processed_chunks, dim=0)
        # --- 修改部分结束 ---

        return audio_feature.flatten(0, 1).split(per_prompt_audio)

    def _postprocess_image_embeds_evs(
        self,
        image_embeds_split: tuple[torch.Tensor, ...],
        image_input: Qwen2_5_VLImageInputs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Append mrope positions for each for images.
        This is necessary to recover correct mrope
        positions after video pruning

        Args:
            image_embeds_split: Tuple of image embeddings for
                each image item.
            image_input: Image input data.

        Returns:
            Tuple of image embeddings for each image item.
            Resulting embeddings will have extra 5 channels for
            computed mrope positions, consistent with video embeddings.
        """
        if self.is_multimodal_pruning_enabled:
            merge_size = self.visual.spatial_merge_size
            grid_thw = image_input["image_grid_thw"]
            grid_thw_list = grid_thw.tolist()
            image_embeds_out = []
            for emb, size in zip(image_embeds_split, grid_thw_list):
                positions = compute_mrope_for_media(size, merge_size).to(emb.device)
                positions = torch.cat(
                    [
                        positions,
                        torch.zeros_like(
                            positions[:, 0:1]
                        ),  # Dummy extra fifth channel
                    ],
                    dim=1,
                )
                emb = torch.cat([emb, positions], dim=1)
                image_embeds_out.append(emb)
            image_embeds_split = tuple(image_embeds_out)
        return image_embeds_split

    def _postprocess_video_embeds_evs(
        self,
        video_embeds_split: tuple[torch.Tensor, ...],
        video_input: Qwen2_5_VLVideoInputs,
    ) -> tuple[torch.Tensor, ...]:
        """
        Prunes video embeddings via Efficient Video Sampling (EVS)
        and then appends mrope positions for each retained embeddings

        Args:
            video_embeds_split: Tuple of video embeddings for each video item.
            video_input: Video input data.

        Returns:
            Tuple of video embeddings for each video item.
            Resulting embeddings will have extra 5 channels for computed mrope
            positions, and whether the index corresponds to a video embedding.
        """
        grid_thw = video_input["video_grid_thw"]
        assert grid_thw.ndim == 2
        grid_thw_list = grid_thw.tolist()
        merge_size = self.visual.spatial_merge_size

        # Apply EVS to each video.
        video_embeds_out = []
        for video_idx, (emb, size) in enumerate(zip(video_embeds_split, grid_thw_list)):
            # Compute positions.
            timestamps = video_input.timestamps[video_idx]
            num_frames = len(timestamps)

            t, h, w = size
            if self.is_multimodal_pruning_enabled:
                # For each video, compute retention mask using EVS.
                # retention_mask: [11424].
                retention_mask = compute_retention_mask(
                    emb,
                    size,
                    spatial_merge_size=self.visual.spatial_merge_size,
                    q=self.video_pruning_rate,
                )
                # Apply retention mask.
                emb = emb[retention_mask]

                # Calculate the actual number of retained tokens per frame.
                num_frames, rows, cols = (
                    t,
                    h // merge_size,
                    w // merge_size,
                )
                retention_mask_thw = retention_mask.reshape(num_frames, rows, cols)
                num_tokens_per_frame = (
                    retention_mask_thw.sum(dim=(1, 2)).long().tolist()
                )
            else:
                feature_size = emb.shape[0] // num_frames
                num_tokens_per_frame = [feature_size] * num_frames
                retention_mask = None

            emb = self._create_final_video_embeddings(
                video_embeddings=emb,
                num_tokens_per_frame=num_tokens_per_frame,
                timestamps=timestamps,
                video_grid_thw=size,
                retention_mask=retention_mask,
            )

            video_embeds_out.append(emb)

        return tuple(video_embeds_out)

    def _create_final_video_embeddings(
        self,
        video_embeddings: torch.Tensor,
        num_tokens_per_frame: list[int],
        timestamps: list[float],
        video_grid_thw: list[int],
        retention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Create final embeddings that combine video embeddings with
        text embeddings of indicator tokens.

        These final embeddings contain:
        - Actual video embeddings in positions corresponding to video content
        - Text embeddings for indicator tokens (<img>, </img>, and
          frame separation text) in their respective positions

        These embeddings will replace the placeholder embeddings to create
        input_embeds for the LLM.
        """
        device = video_embeddings.device

        # Generate video replacement token IDs using get_video_repl
        # This tokenizes each frame separator independently, then uses pre-tokenized
        # special tokens to ensure consistent tokenization regardless of
        # num_tokens_per_frame values.
        video_repl = Qwen3VLMultiModalProcessor.get_video_repl(
            tokens_per_frame=num_tokens_per_frame,
            tokenizer=self._tokenizer,
            timestamps=timestamps,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            video_token_id=self.config.video_token_id,
            select_token_id=self.is_multimodal_pruning_enabled,
        )

        repl_token_ids = torch.tensor(video_repl.full, device=device)
        embed_token_id = _cached_tensor(self.config.video_token_id, device=device)
        is_video_embed = torch.isin(repl_token_ids, embed_token_id)

        # Get text embeddings for indicator tokens (has only `visual_dim``).
        text_embeddings = self.get_language_model().embed_input_ids(repl_token_ids)

        if self.use_deepstack:
            (
                deepstack_input_embeds,
                multimodal_embeddings,
            ) = self._compute_deepstack_embeds(
                inputs_embeds=text_embeddings,
                multimodal_embeddings=[video_embeddings],
                is_multimodal=is_video_embed,
            )
        else:
            deepstack_input_embeds = None
            multimodal_embeddings = [video_embeddings]

        merged_embeddings = _merge_multimodal_embeddings(
            inputs_embeds=text_embeddings,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_video_embed,
        )

        to_concat = [merged_embeddings]
        if deepstack_input_embeds is not None:
            to_concat.append(
                deepstack_input_embeds.permute(1, 0, 2).reshape(
                    deepstack_input_embeds.shape[1], -1
                )
            )

        expanded_positions = None
        if self.is_multimodal_pruning_enabled:
            is_vision_start = repl_token_ids.eq(self.config.vision_start_token_id)
            expanded_positions = self._get_expanded_positions(
                device=merged_embeddings.device,
                seq_len=merged_embeddings.shape[0],
                video_grid_thw=video_grid_thw,
                num_tokens_per_frame=num_tokens_per_frame,
                timestamps=timestamps,
                is_video_embed=is_video_embed,
                is_vision_start=is_vision_start,
                retention_mask=retention_mask,
            )
            to_concat.append(expanded_positions)

        final_video_embeddings = torch.cat(to_concat, dim=-1)

        return final_video_embeddings

    def _get_expanded_positions(
        self,
        device,
        seq_len,
        video_grid_thw,
        num_tokens_per_frame,
        timestamps,
        is_video_embed,
        is_vision_start,
        retention_mask,
    ):
        embed_token_id = _cached_tensor(self.config.video_token_id, device=device)

        # Expand positions to match the full sequence length
        # (includes both video tokens and indicator tokens)
        # Shape: [full_length, 5] where positions are filled for video tokens
        # and zeros for indicator tokens.
        # Channel 3 flags VISION_START tokens so that
        # recompute_mrope_positions can reliably count timestamp tokens
        # (even when early frames have all video tokens pruned).
        # Channel 4 flags video-embedding tokens.
        expanded_positions = torch.zeros(
            seq_len,
            5,  # [t_index, h_index, w_index, is_vision_start, is_video]
            device=device,
            dtype=torch.long,
        )
        _, h, w = video_grid_thw
        merge_size = self.visual.spatial_merge_size
        num_frames = len(num_tokens_per_frame)
        unpruned_token_ids = Qwen3VLMultiModalProcessor.get_video_repl(
            tokens_per_frame=[(h // merge_size) * (w // merge_size)] * num_frames,
            tokenizer=self._tokenizer,
            timestamps=timestamps,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            video_token_id=self.config.video_token_id,
        ).full
        unpruned_token_ids_tensor = torch.tensor(unpruned_token_ids, device=device)
        mm_feature = MultiModalFeatureSpec(
            data=MultiModalKwargsItem(
                {
                    "video_grid_thw": MultiModalFieldElem(
                        data=torch.tensor(video_grid_thw),
                        field=None,  # HACK.
                    ),
                }
            ),
            modality="video",
            identifier="DUMMY",
            mm_position=PlaceholderRange(offset=0, length=len(unpruned_token_ids)),
        )
        original_mrope = (
            self.get_mrope_input_positions(
                input_tokens=unpruned_token_ids,
                mm_features=[mm_feature],
            )[0]
            .to(device)
            .permute(1, 0)
        )
        full_is_video_embed = unpruned_token_ids_tensor == embed_token_id
        expanded_positions[is_video_embed, :3] = original_mrope[full_is_video_embed][
            retention_mask
        ]
        expanded_positions[~is_video_embed, :3] = original_mrope[~full_is_video_embed]
        expanded_positions[..., 3] = is_vision_start
        expanded_positions[..., 4] = is_video_embed

        return expanded_positions

    def _parse_and_validate_multimodal_inputs(self, **kwargs: object) -> dict:
        mm_input_by_modality = {}
        for input_key in kwargs:
            if (
                input_key in ("pixel_values", "image_embeds")
                and "image" not in mm_input_by_modality
            ):
                mm_input_by_modality["image"] = self._parse_and_validate_image_input(
                    **kwargs
                )
            if (
                input_key in ("pixel_values_videos", "video_embeds")
                and "video" not in mm_input_by_modality
            ):
                mm_input_by_modality["video"] = self._parse_and_validate_video_input(
                    **kwargs
                )
            if input_key == "audio_feature" and "audio" not in mm_input_by_modality:
                mm_input_by_modality["audio"] = self._parse_and_validate_audio_input(
                    **kwargs
                )
        # Drop modalities that were not actually present in the inputs.
        mm_input_by_modality = {
            k: v for k, v in mm_input_by_modality.items() if v is not None
        }
        return mm_input_by_modality

    @staticmethod
    def _iter_mm_grid_hw(
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
        video_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
        spatial_merge_size: int,
    ) -> Iterator[tuple[int, int, int, int]]:
        """Iterate over multimodal features and yield position info.

        Args:
            input_tokens: List of token IDs in the input sequence.
            mm_features: List of multimodal feature specifications containing
                image/video data and position information.
            video_token_id: Token ID used for video tokens.
            vision_start_token_id: Token ID marking the start of a vision sequence.
            vision_end_token_id: Token ID marking the end of a vision sequence.
            spatial_merge_size: Size of the spatial merge operation used to
                compute logical grid dimensions from the original feature grid.

        Yields:
            offset: Position of the first video/image token in the sequence.
            llm_grid_h: Logical grid height (may not match actual token count with EVS).
            llm_grid_w: Logical grid width (may not match actual token count with EVS).
            actual_num_tokens: Actual number of video/image tokens in the placeholder.
        """
        for mm_feature in sorted(mm_features, key=lambda f: f.mm_position.offset):
            offset = mm_feature.mm_position.offset
            if mm_feature.modality == "image":
                t, h, w = mm_feature.data["image_grid_thw"].data.tolist()
                assert t == 1, f"Image must have 1 frame, got {t}"
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                yield offset, llm_grid_h, llm_grid_w, llm_grid_h * llm_grid_w
            elif mm_feature.modality == "video":
                if mm_feature.data["video_grid_thw"].data.ndim == 2:
                    for vid in mm_feature.data["video_grid_thw"].data:
                        t, h, w = vid.tolist()
                        llm_grid_h = h // spatial_merge_size
                        llm_grid_w = w // spatial_merge_size

                        for _ in range(t):
                            # When EVS is enabled, some frames may have 0 video tokens in the
                            # placeholder. We use `vision_start_token_id` to locate each frame
                            # since it is always present for every frame.
                            # We then look for the first `video_token_id` after
                            # `vision_start_token_id` and before `vision_end_token_id`.
                            offset = input_tokens.index(vision_start_token_id, offset)
                            vision_end_offset = input_tokens.index(vision_end_token_id, offset)

                            try:
                                video_offset = input_tokens.index(
                                    video_token_id, offset, vision_end_offset
                                )
                                # NOTE: audio pad tokens may be interleaved
                                # inside the vision_start/vision_end block
                                # (use_audio_in_video), so we count the actual
                                # `video_token_id` occurrences instead of using
                                # `vision_end_offset - video_offset`.
                                actual_num_tokens = 0
                                for tok in input_tokens[video_offset:vision_end_offset]:
                                    if tok == video_token_id:
                                        actual_num_tokens += 1
                            except ValueError:
                                # No `video_token_id` in this frame (EVS with 0 tokens for
                                # this frame) -> use `offset + 1`` to move past
                                # `vision_start_token_id`.
                                actual_num_tokens = 0
                                video_offset = offset + 1

                            yield video_offset, llm_grid_h, llm_grid_w, actual_num_tokens
                            # Move offset past this frame for next iteration.
                            offset = vision_end_offset + 1
                else:
                    t, h, w = mm_feature.data["video_grid_thw"].data.tolist()
                    llm_grid_h = h // spatial_merge_size
                    llm_grid_w = w // spatial_merge_size

                    for _ in range(t):
                        # When EVS is enabled, some frames may have 0 video tokens in the
                        # placeholder. We use `vision_start_token_id` to locate each frame
                        # since it is always present for every frame.
                        # We then look for the first `video_token_id` after
                        # `vision_start_token_id` and before `vision_end_token_id`.
                        offset = input_tokens.index(vision_start_token_id, offset)
                        vision_end_offset = input_tokens.index(vision_end_token_id, offset)

                        try:
                            video_offset = input_tokens.index(
                                video_token_id, offset, vision_end_offset
                            )
                            # NOTE: audio pad tokens may be interleaved inside
                            # the vision_start/vision_end block
                            # (use_audio_in_video), so we count the actual
                            # `video_token_id` occurrences instead of using
                            # `vision_end_offset - video_offset`.
                            actual_num_tokens = 0
                            for tok in input_tokens[video_offset:vision_end_offset]:
                                if tok == video_token_id:
                                    actual_num_tokens += 1
                        except ValueError:
                            # No `video_token_id` in this frame (EVS with 0 tokens for
                            # this frame) -> use `offset + 1`` to move past
                            # `vision_start_token_id`.
                            actual_num_tokens = 0
                            video_offset = offset + 1

                        yield video_offset, llm_grid_h, llm_grid_w, actual_num_tokens
                        # Move offset past this frame for next iteration.
                        offset = vision_end_offset + 1
            elif mm_feature.modality == "audio":
                # Audio is an independent, interleaved-in-video modality whose
                # pad tokens live inside the video's vision block. For mrope,
                # audio pad tokens occupy regular (text) positions (matching the
                # training-time rope layout in data_qwen.py), so we do not yield
                # any vision grid for them.
                continue
            else:
                raise ValueError(f"Unsupported modality: {mm_feature.modality}")

    def _get_evs_mask_segments(
        self, mm_position: PlaceholderRange, expected_frames: int
    ) -> list[torch.Tensor] | None:
        """Extract contiguous segments from EVS is_embed mask.

        The EVS (Efficient Video Sampling) mask marks which placeholder
        positions should be filled with video embeddings. This method splits
        the mask into contiguous segments, where each segment represents one
        retained frame.

        This is a pure function - it does not modify any state and always
        returns the same output for the same input (idempotent).

        Args:
            mm_position: MultiModal position containing the is_embed mask
            expected_frames: Expected number of frame segments

        Returns:
            List of tensors, each containing indices for one frame segment,
            or None if EVS is not enabled or validation fails.
        """
        is_embed_mask = getattr(mm_position, "is_embed", None)
        if is_embed_mask is None:
            return None

        # Find all True positions in the mask
        mask_tensor = torch.as_tensor(is_embed_mask, dtype=torch.bool).view(-1)
        true_indices = torch.nonzero(mask_tensor, as_tuple=False).flatten()
        if true_indices.numel() == 0:
            return None

        # Split into contiguous segments (where diff > 1 indicates a gap)
        if true_indices.numel() == 1:
            segments = [true_indices]
        else:
            diffs = torch.diff(true_indices)
            split_points = torch.nonzero(diffs != 1, as_tuple=False).flatten()
            if split_points.numel() == 0:
                segments = [true_indices]
            else:
                segments = torch.tensor_split(
                    true_indices, split_points.add(1).tolist()
                )

        # Validate segment count matches expected frames
        if len(segments) < expected_frames:
            logger.debug(
                "EVS mask segments (%d) do not match expected frames (%d)",
                len(segments),
                expected_frames,
            )
            return None

        return segments[:expected_frames]

    def _extract_frame_offsets_from_mask(
        self, mm_position: PlaceholderRange, expected_frames: int
    ) -> list[int] | None:
        """Return relative offsets for each EVS-retained frame.

        The prompt processor stores a boolean mask inside ``mm_position`` that
        marks which placeholder locations should be populated with video
        embeddings. By splitting that mask into contiguous runs we can recover
        the start of every retained frame without probing ``input_tokens``.

        Args:
            mm_position: MultiModal position containing the is_embed mask
            expected_frames: Expected number of frames

        Returns:
            List of starting offsets (relative to mm_position) for each frame,
            or None if EVS is not enabled.
        """
        segments = self._get_evs_mask_segments(mm_position, expected_frames)
        if segments is None:
            return None

        return [int(segment[0].item()) for segment in segments]

    def _get_actual_frame_token_counts(
        self, mm_position: PlaceholderRange, expected_frames: int
    ) -> list[int] | None:
        """Return actual token count for each EVS-retained frame.

        This function calculates the actual number of tokens per frame by
        analyzing the is_embed mask, accounting for EVS pruning. Each frame
        may have a different token count due to content-aware pruning.

        Args:
            mm_position: MultiModal position containing the is_embed mask
            expected_frames: Expected number of frames

        Returns:
            List of token counts for each frame, or None if EVS is not enabled.
        """
        segments = self._get_evs_mask_segments(mm_position, expected_frames)
        if segments is None:
            return None

        return [len(seg) for seg in segments]

    def get_mrope_input_positions(
        self,
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
    ) -> tuple[torch.Tensor, int]:
        return self._get_mrope_input_positions(
            input_tokens=input_tokens,
            mm_features=mm_features,
            config=self.config,
        )

    @staticmethod
    def _get_mrope_input_positions(
        input_tokens: list[int],
        mm_features: list[MultiModalFeatureSpec],
        config: LocalQwen3VLConfig,
    ):
        llm_pos_ids_list = []
        st = 0
        for (
            offset,
            llm_grid_h,
            llm_grid_w,
            actual_num_tokens,
        ) in Qwen3VLForConditionalGeneration._iter_mm_grid_hw(
            input_tokens,
            mm_features,
            video_token_id=config.video_token_id,
            vision_start_token_id=config.vision_start_token_id,
            vision_end_token_id=config.vision_end_token_id,
            spatial_merge_size=config.vision_config.spatial_merge_size,
        ):
            # Skip frames with 0 tokens (EVS placeholder with tokens lumped elsewhere)
            if actual_num_tokens == 0:
                continue

            text_len = offset - st
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

            # Check if this is a "lumped placeholder" (all tokens from multiple frames
            # assigned to the 0-th frame - see
            # `Qwen3VLMultiModalProcessor.get_video_repl`.
            expected_tokens_per_frame = llm_grid_h * llm_grid_w
            if actual_num_tokens > expected_tokens_per_frame:
                # Lumped placeholder: create grid positions for all "logical" frames
                # represented.
                num_logical_frames = actual_num_tokens // expected_tokens_per_frame
                remainder = actual_num_tokens % expected_tokens_per_frame

                # Create positions for complete frames.
                for _ in range(num_logical_frames):
                    grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(
                        3, -1
                    )
                    llm_pos_ids_list.append(grid_indices + text_len + st_idx)
                    st_idx = llm_pos_ids_list[-1].max() + 1
                    text_len = 0  # No text between frames within the lump

                # Handle remainder tokens if any (partial frame).
                # NOTE: this should never be the case. Should we have an assert?
                if remainder > 0:
                    # Create a partial grid - take first 'remainder' positions
                    full_grid = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                    grid_indices = full_grid[:, :remainder]
                    llm_pos_ids_list.append(grid_indices + text_len + st_idx)
            else:
                # Normal case: frame has exactly the expected tokens (after actual EVS
                # pruning).
                grid_indices = np.indices((1, llm_grid_h, llm_grid_w)).reshape(3, -1)
                llm_pos_ids_list.append(grid_indices + text_len + st_idx)

            st = offset + actual_num_tokens

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len), (3, text_len)) + st_idx
            )

        llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        return torch.from_numpy(llm_positions), mrope_position_delta

    def recompute_mrope_positions(
        self,
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
    ) -> tuple[MultiModalEmbeddings, torch.Tensor, int]:
        """
        Update part of input mrope positions (starting with
        num_computed_tokens index). Original mrope_positions are computed
        for unpruned sequence and becomes incorrect once pruning occurs,
        so once we prune media tokens we should reflect this in the
        mrope_positions before we feed it to LLM.

        Args:
            input_ids: (N,) All input tokens of the prompt containing
                entire sequence.
            multimodal_embeddings: Tuple of multimodal embeddings that
                fits into the prefill chunk that is being processed.
            mrope_positions: Existing mrope positions (3, N) for entire
                sequence
            num_computed_tokens: A number of computed tokens so far.

        Returns:
            Tuple of (multimodal_embeddings, mrope_positions,
                mrope_position_delta).
        """
        return self._recompute_mrope_positions(
            input_ids=input_ids,
            multimodal_embeddings=multimodal_embeddings,
            mrope_positions=mrope_positions,
            num_computed_tokens=num_computed_tokens,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )

    @staticmethod
    def _recompute_mrope_positions(
        input_ids: list[int],
        multimodal_embeddings: MultiModalEmbeddings,
        mrope_positions: torch.LongTensor,
        num_computed_tokens: int,
        vision_start_token_id: int,
        image_token_id: int,
        video_token_id: int,
    ) -> tuple[MultiModalEmbeddings, torch.Tensor, int]:
        # Device
        device = (
            multimodal_embeddings[0].device
            if len(multimodal_embeddings)
            else mrope_positions.device
        )

        # Tensors
        input_ids_t = torch.as_tensor(input_ids, device=device, dtype=torch.long)

        mm_embeddings_out = []
        mm_embeddings_pos = []
        # Strip position information from embeddings (last 5 channels)
        # For Qwen3 VL, handle potentially empty frames (from unpacking)
        for mm in multimodal_embeddings:
            if mm.shape[0] > 0:  # Only process non-empty frames
                mm_embeddings_out.append(mm[:, :-5])
                mm_embeddings_pos.append(mm[:, -5:].permute(1, 0).long())
            else:
                # Empty frame - keep as is
                mm_embeddings_out.append(mm)
                # Create empty position tensor with correct shape
                mm_embeddings_pos.append(
                    torch.empty(5, 0, device=device, dtype=torch.long)
                )

        positions, mrope_positions_delta = recompute_mrope_positions(
            input_ids_t,
            mm_embeddings_pos,
            mrope_positions,
            num_computed_tokens,
            vision_start_token_id,
            image_token_id,
            video_token_id,
        )

        return tuple(mm_embeddings_out), positions, mrope_positions_delta

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings | None:
        mm_input_by_modality = self._parse_and_validate_multimodal_inputs(**kwargs)
        if not mm_input_by_modality:
            return None

        # The result multimodal_embeddings is a flat tuple of tensors, with each
        # tensor corresponding to a single multimodal data item (image, video or
        # audio). Video and audio are independent modalities/features even when
        # their placeholder ranges overlap (use_audio_in_video), so we return
        # them as separate per-item tensors and let the runner slice them per
        # chunk. Ordering matches the modality iteration order.
        multimodal_embeddings: list[torch.Tensor] = []

        # NOTE: It is important to iterate over the keys in this dictionary
        # to preserve the order of the modalities.
        for modality in mm_input_by_modality:
            multimodal_input = mm_input_by_modality[modality]
            if modality == "image":
                image_embeddings = self._process_image_input(multimodal_input)
                image_embeddings = self._postprocess_image_embeds_evs(
                    image_embeddings, multimodal_input
                )
                multimodal_embeddings.extend(image_embeddings)
            if modality == "video":
                video_embeddings = self._process_video_input(multimodal_input)
                if self.is_multimodal_pruning_enabled:
                    video_embeddings = self._postprocess_video_embeds_evs(
                        video_embeddings, multimodal_input
                    )
                multimodal_embeddings.extend(video_embeddings)
            if modality == "audio":
                audio_embeddings = self._process_audio_input(multimodal_input)
                multimodal_embeddings.extend(audio_embeddings)

        return tuple(multimodal_embeddings)

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        handle_oov_mm_token: bool = False,
    ) -> torch.Tensor:
        image_mask = (input_ids == self.config.image_token_id) & is_multimodal
        video_mask = (input_ids == self.config.video_token_id) & is_multimodal
        audio_mask = (input_ids == self.config.audio_token_id) & is_multimodal
        vision_mask = image_mask | video_mask
        is_multimodal = image_mask | video_mask | audio_mask
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.language_model.embed_input_ids,
            is_multimodal=is_multimodal,
            handle_oov_mm_token=handle_oov_mm_token,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        # multimodal_embeddings is a flat tuple of per-item tensors already
        # sliced to the current (possibly chunked) prefill window by the
        # runner. Vision (image/video) items carry deepstack channels
        # (feature dim == visual_dim + multiscale_dim) while audio items carry
        # the plain text hidden dim, so they are distinguishable by shape.
        multimodal_embeddings = list(multimodal_embeddings)

        # Separate vision (image/video) items from audio items. Vision items
        # carry deepstack channels (feature dim == visual_dim + multiscale_dim)
        # so they are identifiable by shape; every other item is audio. NOTE:
        # audio embeddings have feature dim == visual_dim (they are aligned to
        # the text/visual hidden size), so we must NOT treat "plain visual_dim"
        # as vision - only the presence of deepstack channels marks vision.
        # Video and audio are then scattered against their OWN masks
        # (vision_mask / audio_mask), never against a combined mask, matching
        # the reference implementation and staying correct regardless of how
        # video/audio pads are interleaved in the prompt.
        vision_embeds: list[torch.Tensor] = []
        multiscale_embeds: list[torch.Tensor] = []
        audio_embeds: list[torch.Tensor] = []
        for embeddings in multimodal_embeddings:
            if (
                self.use_deepstack
                and embeddings.shape[-1] == self.visual_dim + self.multiscale_dim
            ):
                embeddings_main, embeddings_multiscale = torch.split(
                    embeddings, [self.visual_dim, self.multiscale_dim], dim=-1
                )
                vision_embeds.append(embeddings_main)
                multiscale_embeds.append(embeddings_multiscale)
            elif not self.use_deepstack and embeddings.shape[-1] == self.visual_dim:
                # Without deepstack, both vision and audio carry visual_dim, so
                # we cannot distinguish them by shape. Fall back to counting
                # against masks below.
                vision_embeds.append(embeddings)
            else:
                audio_embeds.append(embeddings)

        if self.use_deepstack and multiscale_embeds:
            deepstack_input_embeds = inputs_embeds.new_zeros(
                inputs_embeds.size(0), self.multiscale_dim
            )
            deepstack_input_embeds = _merge_multimodal_embeddings(
                inputs_embeds=deepstack_input_embeds,
                multimodal_embeddings=multiscale_embeds,
                is_multimodal=vision_mask,
            )
            deepstack_input_embeds = (
                deepstack_input_embeds.view(
                    inputs_embeds.shape[0], self.deepstack_num_level, self.visual_dim
                )
                .permute(1, 0, 2)
                .contiguous()
            )
            self._set_deepstack_input_embeds(deepstack_input_embeds)

        if vision_embeds:
            inputs_embeds = _merge_multimodal_embeddings(
                inputs_embeds=inputs_embeds,
                multimodal_embeddings=vision_embeds,
                is_multimodal=vision_mask,
            )
        if audio_embeds:
            audio_positions = audio_mask.nonzero(as_tuple=True)[0]
            inputs_embeds[audio_positions] = torch.cat(audio_embeds, dim=0).to(
                inputs_embeds.dtype
            )
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        """Run forward pass for Qwen3VL.

        Args:
            input_ids: Flattened (concatenated) input_ids corresponding to a
                batch.
            positions: Flattened (concatenated) position ids corresponding to a
                batch.
                **NOTE**: If mrope is enabled (default setting for Qwen3VL
                opensource models), the shape will be `(3, seq_len)`,
                otherwise it will be `(seq_len,).
            intermediate_tensors: Intermediate tensors from previous pipeline
                stages.
            inputs_embeds: Pre-computed input embeddings.
            **kwargs: Additional keyword arguments including:
                - pixel_values: Pixel values to be fed to a model.
                    `None` if no images are passed.
                - image_grid_thw: Tensor `(n_images, 3)` of image 3D grid in
                    LLM. `None` if no images are passed.
                - pixel_values_videos: Pixel values of videos to be fed to a
                    model. `None` if no videos are passed.
                - video_grid_thw: Tensor `(n_videos, 3)` of video 3D grid in
                    LLM. `None` if no videos are passed.
        """

        if intermediate_tensors is not None:
            inputs_embeds = None

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            deepstack_input_embeds = self._get_deepstack_input_embeds(
                inputs_embeds.size(0)
            )
        else:
            deepstack_input_embeds = None

        hidden_states = self.language_model.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            # args for deepstack
            deepstack_input_embeds=deepstack_input_embeds,
        )

        if inputs_embeds is not None and get_pp_group().is_first_rank:
            self._clear_deepstack_input_embeds(inputs_embeds.size(0))

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        filtered_weights = (
            (name, weight)
            for name, weight in weights
            if "audio" not in name or "position_ids" not in name
        )
        return loader.load_weights(filtered_weights, mapper=self.hf_to_vllm_mapper)

    def get_mm_mapping(self) -> MultiModelKeys:
        """
        Get the module prefix in multimodal models
        """
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            connector=["visual.merger", "visual.deepstack_merger_list"],
            tower_model="visual.",
        )

    def get_num_mm_encoder_tokens(
        self,
        num_image_tokens: int,
    ) -> int:
        hf_config = self.config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size

        return num_image_tokens * merge_size**2

    def get_num_mm_connector_tokens(
        self,
        num_vision_tokens: int,
    ) -> int:
        hf_config = self.config
        vision_config = hf_config.vision_config
        merge_size = vision_config.spatial_merge_size
        return num_vision_tokens // merge_size**2


@lru_cache
def _cached_tensor(x, device) -> torch.Tensor:
    return torch.tensor(x, device=device)
