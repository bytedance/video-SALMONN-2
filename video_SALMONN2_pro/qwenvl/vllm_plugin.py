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


def register() -> None:
    from vllm.model_executor.models import ModelRegistry

    # Importing this module also runs `_register_qwen3vl_video_backend()`,
    # which registers the "qwen3vl" VideoLoader backend.
    from qwenvl.model.vllm_qwen3 import (
        Qwen3VLForConditionalGeneration as VLLMQwen3VLForConditionalGeneration,
    )

    ModelRegistry.register_model(
        "Qwen3VLForConditionalGeneration",
        VLLMQwen3VLForConditionalGeneration,
    )
    ModelRegistry.register_model(
        "qwen3-vl",
        VLLMQwen3VLForConditionalGeneration,
    )
