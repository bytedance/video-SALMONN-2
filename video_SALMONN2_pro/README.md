# video-SALMONN 2 Pro: another updated version of video-SALMONN 2

video-SALMONN 2 Pro upgrades video-SALMONN 2 to the Qwen3-VL backbone while
retaining its audio-visual alignment and instruction-tuning pipeline. The
project supports joint video and audio understanding, detailed video
captioning, and audio-visual question answering.

Three model sizes are provided: 4B, 8B, and 32B.

<div style="display:flex; gap:0.25rem;">
<a href="https://arxiv.org/abs/2506.15220"><img src="https://img.shields.io/badge/paper-PDF-green"></a>
<a href="https://huggingface.co/tsinghua-ee/video-SALMONN-2-Pro-4B"><img src="https://img.shields.io/badge/video--SALMONN_2_Pro-4B-yellow"></a>
<a href="https://huggingface.co/tsinghua-ee/video-SALMONN-2-Pro-8B"><img src="https://img.shields.io/badge/video--SALMONN_2_Pro-8B-yellow"></a>
<a href="https://huggingface.co/tsinghua-ee/video-SALMONN-2-Pro-32B"><img src="https://img.shields.io/badge/video--SALMONN_2_Pro-32B-yellow"></a>
</div>

## Performance

We evaluate the models on audio-visual QA benchmarks including Video-MME, WorldSense, AVUT, Video-Holmes, and DailyOmni. Our 4B and 8B models achieve SOTA results at comparable scales, while the 32B model surpasses all other open-source systems.

| Model | Video-MME | WorldSense | AVUT | Video-Holmes | DailyOmni | FutureOmni |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VideoLLaMA3 (2B) | 59.6 | - | - | - | - | - |
| Qwen2.5-Omni (3B)$^*$ | 62.0 | - | - | - | 40.5 | 38.9 |
| Qwen2.5-VL (3B) | 61.5 | - | - | - | 37.4 | - |
| **video-SALMONN 2+ (3B)$^*$** | 68.3 | 48.3 | 66.2 | 42.2 | 67.7 | 50.5 |
| **video-SALMONN 2 Pro (4B)$^*$** | **75.4** | **54.3** | **77.2** | **50.1** | **78.9** | **55.6** |
||||||||
| video-SALMONN (13B)$^*$ | 43.3 | - | 38.3 | - | - | - |
| LLaVA-Video (7B) | 63.3 | 40.2 | 56.5 | - | - | - |
| VideoLLaMA2 (7B)$^*$ | 54.9 | 25.4 | 44.9 | - | 35.2 | 40.8 |
| VideoLLaMA3 (7B) | 66.2 | - | - | - | - | 46.8 |
| Qwen2.5-Omni (7B)$^*$ | 64.3 | 45.4 | - | 16.4 | 47.5 | 47.5 |
| Qwen2.5-VL (7B) | 65.1 | - | - | 27.8 | 40.7 | 43.7 |
| **video-SALMONN 2 (7B)$^*$** | 67.4 | 48.6 | 65.6 | 40.7 | 66.3 | 48.7 |
| **video-SALMONN 2+ (7B)$^*$** | 73.4 | 50.9 | 69.5 | 46.9 | 71.8 | 51.0 |
| **video-SALMONN 2 Pro (8B)$^*$** | **77.5** | **55.0** | **78.3** | **55.1** | **80.0** | **58.0** |
||||||||
| GPT-4o | 71.9 | 42.6 | 56.6 | 42.0 | 56.5 | 49.7 |
| Gemini-1.5 Pro$^*$ | 75.0 | 48.0 | 78.3 | 41.2 | - | - |
| Qwen3-Omni-Flash$^*$ | 71.4 | 54.1 | - | 57.3 | 76.2 | - |
| LLaVA-Video (72B) | 70.5 | - | - | - | - | - |
| VideoLLaMA2 (72B)$^*$ | 61.4 | - | - | - | - | - |
| Qwen2.5-VL (72B) | 73.3 | - | - | 50.2 | 61.8 | - |
| **video-SALMONN 2+ (72B)$^*$** | 79.7 | 56.5 | 72.2 | 57.8 | 79.4 | 60.8 |
| **video-SALMONN 2 Pro (32B)$^*$** | **82.7** | **59.4** | **81.1** | **61.7** | **81.7** | **61.4** |

$^*$ Models that use audio input.

## Installation

Install the project dependencies:

```bash
pip install -r requirements.txt
```

The inference and training paths require PyTorch, Transformers, and a working
video decoding backend. GPU inference requires a CUDA-compatible PyTorch
installation.

## Inference

Use `scripts/inference.py` for single-video inference:

```bash
python scripts/inference.py \
  --model tsinghua-ee/video-SALMONN-2-Pro-8B \
  --video dataset/videos/example.mp4
```

Use a custom instruction with `--prompt`:

```bash
python scripts/inference.py \
  --model tsinghua-ee/video-SALMONN-2-Pro-4B \
  --video dataset/videos/example.mp4 \
  --prompt "Describe the video and summarize its spoken content."
```

The same command works for the 4B, 8B, and 32B checkpoints. Use `--no-audio`
when embedded audio should not be processed. Frame sampling and generation can
be adjusted with `--video-min-frames`, `--video-max-frames`,
`--video-max-frame-pixels`, and `--max-new-tokens`.

## Data Format

Training data is stored as JSON or JSONL. A video sample follows this format:

```json
[
  {
    "video": "dataset/videos/example.mp4",
    "use_audio": true,
    "conversations": [
      {
        "from": "human",
        "value": "<video>\nDescribe the video in detail."
      },
      {
        "from": "gpt",
        "value": "The expected response."
      }
    ]
  }
]
```

Set `use_audio` to `false` when the video audio track should be ignored.

## Training

`scripts/train.sh` launches supervised fine-tuning. It loads the checkpoint
directly from `--model_base` and writes checkpoints to `output/<run_name>`.

Set the distributed launch environment before running the script:

```bash
export ARNOLD_WORKER_GPU=8
export ARNOLD_WORKER_NUM=1
export ARNOLD_ID=0
export METIS_WORKER_0_HOST=127.0.0.1
```

Audio alignment:

```bash
bash scripts/train.sh \
  --model_base tsinghua-ee/video-SALMONN-2-Pro-8B \
  --dataset dataset/audio_alignment.json \
  --run_name audio_alignment \
  --interval 0.1 \
  --max_frames 768 \
  --max_pixels 61250 \
  --train_qformer \
  --lr 2e-5 \
  --epoch 5
```

Audio-visual supervised fine-tuning with LoRA:

```bash
bash scripts/train.sh \
  --model_base tsinghua-ee/video-SALMONN-2-Pro-8B \
  --dataset dataset/av_sft.json \
  --run_name av_sft \
  --interval 0.1 \
  --max_frames 768 \
  --max_pixels 61250 \
  --train_qformer \
  --train_proj \
  --use_lora \
  --lora_r 128 \
  --lora_alpha 256 \
  --lr 2e-5 \
  --epoch 5
```

Adjust per-device batch size and gradient accumulation with `--bs` and
`--accum_steps`. The 32B model normally requires more GPUs or a more aggressive
DeepSpeed configuration.

## Merge a LoRA Checkpoint

LoRA merging is intentionally separate from training. Merge a trained adapter
into its base model with:

```bash
python scripts/merge_base_model.py \
  --model_name_or_path tsinghua-ee/video-SALMONN-2-Pro-8B \
  --lora_ckpt output/av_sft/checkpoint-1000 \
  --output_dir output/av_sft-merged
```

The merged directory can be passed directly to `scripts/inference.py`.

## vLLM Serving

An OpenAI-compatible vLLM server is also available:

```bash
python scripts/serve.py \
  --model tsinghua-ee/video-SALMONN-2-Pro-8B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name qwen3-vl
```

Send a request with:

```bash
python scripts/call_vllm.py \
  --video dataset/videos/example.mp4 \
  --prompt "Describe the video in detail."
```
