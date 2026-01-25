# video-SALMONN 2+ — Transformers-Compatible Usage

This folder adds a Transformers-friendly entry point for the video-SALMONN 2+ model so you can use
`AutoModelForCausalLM`, `AutoProcessor`, Trainer, and `generate()` without custom scripts.

## What’s Included

- **Config**: `Qwen2_5_VLConfig` now carries multimodal token ids/strings and `auto_map` for Auto* loading.
- **Model Alias**: `VideoSALMONN2PlusForConditionalGeneration` (alias of `video_SALMONN2_plus`) for standard naming.
- **Processor**: `VideoSALMONN2PlusProcessor` handles image/video/audio inputs and expands placeholders.
- **Optional Registration**: `register_transformers()` registers AutoConfig/AutoModel/AutoProcessor locally.

## Load with Auto* (recommended)

This requires `trust_remote_code=True` when loading from a repo containing this code.

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
    "path_or_hub_repo",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("path_or_hub_repo", trust_remote_code=True)
```

## Local Auto* Registration (no trust_remote_code)

```python
from qwenvl.model import register_transformers
register_transformers()
```

Then load as usual (without `trust_remote_code`).

## Inference Examples

### Video + Audio

```python
inputs = processor(
    text="Describe this video and its audio: <video>",
    videos=video_frames,      # list/np/torch frames
    audio=audio_waveform,     # raw waveform array
    return_tensors="pt",
)
generated = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(generated, skip_special_tokens=True)[0])
```

### Image Only

```python
inputs = processor(
    text="What is in this image? <image>",
    images=image,
    return_tensors="pt",
)
generated = model.generate(**inputs, max_new_tokens=64)
```

### Audio Only

```python
inputs = processor(
    text="Transcribe and summarize this audio: <audio>",
    audio=audio_waveform,
    return_tensors="pt",
)
generated = model.generate(**inputs, max_new_tokens=128)
```

## Placeholder Rules

You may use `<image>`, `<video>`, `<audio>` placeholders in the prompt. The processor expands them into
`<|vision_start|><|*_pad|>...<|vision_end|>` sequences based on the actual input sizes.

If you already use `<|image_pad|>`, `<|video_pad|>`, or `<|audio_pad|>`, you can place a **single** token
per input; the processor will expand it to the correct length.

## Audio Notes

- Pass raw waveform arrays (not file paths).
- Default sampling rate: **16 kHz**.
- Audio is chunked into 30-second windows; each window corresponds to 60 audio tokens.
- For video + audio, the processor interleaves audio tokens per video timestep.

## Files Added/Updated

- `qwenvl/model/processing_video_salmonn2_plus.py`
- `qwenvl/model/configuration_qwen2_5_vl.py`
- `qwenvl/model/modeling_qwen2_5_vl.py`
- `qwenvl/model/__init__.py`
- `video_SALMONN2_plus/README.md` (Auto* usage snippet)
- `video_SALMONN2_plus/README_transformers.md` (this file)
