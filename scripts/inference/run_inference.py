import os
import sys
import yaml
import torch
import transformers

# Add video-SALMONN-2 directory to Python path
salmonn_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if salmonn_root not in sys.path:
    sys.path.insert(0, salmonn_root)

from llava.model import VideoSALMONN2ForCausalLM
from llava.dataset import make_test_data_module
from llava import conversation as conversation_lib
from llava.mm_utils import KeywordsStoppingCriteria
from llava.train.train import ModelArguments, DataArguments, TrainingArguments

def create_conversation_data(question, video_path=None, audio_path=None):
    """Create conversation structure for the dataset."""
    conversation = [
        {"from": "human", "value": ("<image>\n" if video_path else "") + question.strip()},
        {"from": "gpt", "value": ""}
    ]
    
    data = {"conversations": conversation}
    if video_path:
        data["video"] = video_path
    if audio_path:
        data["audio"] = audio_path
    
    return data

def move_to_device(batch, device, text_only=False):
    """Move batch tensors to device."""
    for k in ["input_ids", "labels", "attention_mask"]:
        batch[k] = batch[k].to(device)
    
    if not text_only:
        batch["images"] = [img.to(torch.bfloat16).to(device) for img in batch["images"]]
        batch["spectrogram"] = batch["spectrogram"].to(torch.bfloat16).to(device)
    
    return batch

def run_demo(yaml_file, model_ckpt, device="cuda", bf16=True):
    # Load config
    with open(yaml_file, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create arguments
    model_args = ModelArguments(
        version="qwen_1_5",
        audio_visual=True,
        add_time_token=True,
        mm_pooling_position="after"
    )
    
    data_args = DataArguments(
        video_fps=cfg.get("fps", 1),
        audio_processor="openai/whisper-large-v3",
        max_time=cfg.get("max_time", 110)
    )
    data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    data_args.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    
    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_ckpt, padding_side="right")
    cfg_pretrained = transformers.AutoConfig.from_pretrained(model_ckpt)
    
    # Configure model
    cfg_pretrained.model_args = vars(model_args)
    cfg_pretrained.add_time_token = True
    
    model = VideoSALMONN2ForCausalLM.from_pretrained(
        model_ckpt,
        config=cfg_pretrained,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation="eager",
        audio_visual=True,
        video_fps=data_args.video_fps,
        whisper_path=data_args.audio_processor,
    ).to(device).eval()
    
    model.tokenizer = tokenizer
    
    # Set model config (consolidated)
    config_updates = {
        'mm_use_im_start_end': model_args.mm_use_im_start_end,
        'mm_use_im_patch_token': model_args.mm_use_im_patch_token,
        'mm_pooling_position': model_args.mm_pooling_position,
        'mm_spatial_pool_stride': 2,
        'mm_spatial_pool_mode': "average",
        'mm_newline_position': "grid",
        'modality_max_length': "None"
    }
    for k, v in config_updates.items():
        setattr(model.config, k, v)
    
    # Setup data processing
    data_args.image_processor = model.get_vision_tower().image_processor
    data_module = make_test_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Create test data
    text_only = cfg.get("text_only", False)
    conversation_data = create_conversation_data(
        cfg["question"],
        None if text_only else cfg["video_path"],
        cfg.get("audio_path")
    )
    
    data_module["eval_dataset"].list_data_dict = [conversation_data]
    item = data_module["eval_dataset"]._get_item(0)
    batch = data_module["data_collator"]([item])
    batch = move_to_device(batch, device, text_only)
    
    # Generate
    conv = conversation_lib.conv_templates["qwen_1_5"].copy()
    stopping = KeywordsStoppingCriteria([conv.sep], tokenizer, batch["input_ids"])
    
    result = model.generate(
        **{k: v for k, v in batch.items() if k not in ["ids", "prompts", "ce_only", "texts", "ori_item"]},
        do_sample=cfg.get("do_sample", False),
        top_p=cfg.get("top_p", 0.9),
        max_new_tokens=cfg.get("max_new_tokens", 1024),
        stopping_criteria=[stopping],
    )
    
    print("\n=== Result ===")
    print(tokenizer.decode(result[0].tolist(), skip_special_tokens=True))

if __name__ == "__main__":
    yaml_file = "/opt/txt2img/txt2img/captioner/video-SALMONN-2/scripts/inference/my_inference.yaml"
    model_ckpt = "tsinghua-ee/video-SALMONN-2"
    run_demo(yaml_file, model_ckpt)
