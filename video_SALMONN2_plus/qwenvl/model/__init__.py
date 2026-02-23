from .configuration_qwen2_5_vl import BertConfig, Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig, WhisperConfig
from .modeling_qwen2_5_vl import VideoSALMONN2PlusForConditionalGeneration, video_SALMONN2_plus
from .processing_video_salmonn2_plus import VideoSALMONN2PlusProcessor


def register_transformers():
    """
    Optional: register the VideoSALMONN2Plus classes with Transformers Auto* APIs.
    Call this if you want to use AutoModel/AutoProcessor without trust_remote_code.
    """
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    AutoConfig.register("qwen2_5_vl", Qwen2_5_VLConfig, exist_ok=True)
    AutoModelForCausalLM.register(Qwen2_5_VLConfig, VideoSALMONN2PlusForConditionalGeneration, exist_ok=True)
    AutoProcessor.register(Qwen2_5_VLConfig, VideoSALMONN2PlusProcessor, exist_ok=True)


__all__ = [
    "BertConfig",
    "Qwen2_5_VLConfig",
    "Qwen2_5_VLVisionConfig",
    "WhisperConfig",
    "video_SALMONN2_plus",
    "VideoSALMONN2PlusForConditionalGeneration",
    "VideoSALMONN2PlusProcessor",
    "register_transformers",
]
