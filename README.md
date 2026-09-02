# video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language Models

🚀🚀 Welcome to the repo of **video-SALMONN 2**!

video-SALMONN 2 is a powerful audio-visual large language model (LLM) that **generates high-quality audio-visual video captions**, which is developed by the Department of Electronic Engineering at Tsinghua University and ByteDance. 

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2506.15220'><img src='https://img.shields.io/badge/video_SALMONN_2_paper-PDF-green'></a>
<a href='https://video-salmonn-2.github.io'><img src='https://img.shields.io/badge/Demo-link-green'></a>
<a href="https://arxiv.org/abs/2506.15220"><img src="https://img.shields.io/badge/paper-PDF-green"></a>
<a href="https://huggingface.co/tsinghua-ee/video-SALMONN-2-Pro-4B"><img src="https://img.shields.io/badge/video--SALMONN_2_Pro-4B-yellow"></a>
<a href="https://huggingface.co/tsinghua-ee/video-SALMONN-2-Pro-8B"><img src="https://img.shields.io/badge/video--SALMONN_2_Pro-8B-yellow"></a>
<a href="https://huggingface.co/tsinghua-ee/video-SALMONN-2-Pro-32B"><img src="https://img.shields.io/badge/video--SALMONN_2_Pro-32B-yellow"></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN2_plus_3B_full'><img src='https://img.shields.io/badge/video_SALMONN_2+_3B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN2_plus_7B_full'><img src='https://img.shields.io/badge/video_SALMONN_2+_7B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_72B'><img src='https://img.shields.io/badge/video_SALMONN_2+_72B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2'><img src='https://img.shields.io/badge/video_SALMONN_2_7B-checkpoint-yellow'></a>
<a href='https://huggingface.co/datasets/tsinghua-ee/video-SALMONN_2_testset'><img src='https://img.shields.io/badge/video_SALMONN_2-testset-yellow'></a>
</div>

## 🔥 News

-  **2026-09-03**: We release the code and checkpoint of video-SALMONN 2 pro 4B, 8B, and 32B. Based on Qwen3-VL backbone, the model further improves on audio-visual QA benchmarks. With the support of vLLM, video-SALMONN 2 significantly improves caption generation throughput and latency. 
-  **2026-02-24**: We release the minimal inference code for video-SALMONN 2+ 3B and 7B.
-  **2026-02-15**: We release audio-aligned video-SALMONN 2+ 3B checkpoint.
-  **2026-01-28**: We release the audio-aligned model of video-SALMONN 2+ 72B for finetuning larger audio-visual models.
-  **2025-12-18**: We have released the audio-aligned model of video-SALMONN 2+ 7B for further finetuning. 
-  **2025-09-26**: A new version (Version-2509) of video-SALMONN 2+ is released, containing minor [code](https://github.com/bytedance/video-SALMONN-2/tree/main/video_SALMONN2_plus) revision, an update for [7B model](https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B) and [72B model](https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_72B), as well as the addition of [3B model](https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_3B). The upgraded video-SALMONN 2+ further enhances audio-visual and visual-only understanding capability on various benchmarks. 
- **2025-07-17**: We release the code and checkpoint of video-SALMONN 2+ at [video-SALMONN 2+](https://github.com/bytedance/video-SALMONN-2/tree/main/video_SALMONN2_plus) （Version-2507). video-SALMONN 2+ achieves SOTA results on [Video-MME](https://video-mme.github.io/home_page.html) benchmark.
- **2025-07-08**: We release the 7B version of video-SALMONN 2.
- **2025-06-18**: We release the code of video-SALMONN 2.

## ⚡️ Results

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


## 🌈 How to Use

### For video-SALMONN 2+, please refer to [video_SALMONN2_plus](https://github.com/bytedance/video-SALMONN-2/tree/main/video_SALMONN2_plus)
### For video-SALMONN 2 Pro, please refer to [video_SALMONN2_pro](https://github.com/bytedance/video-SALMONN-2/tree/main/video_SALMONN2_pro)

### How to train video-SALMONN 2

1. Prepare the dataset following `scripts/example_sft.json` and `scripts/example_dpo.json`.
2. Download LLaVA-OneVision Model from [huggingface](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov).
3. Modify the parameters in `scripts/train_sft.sh` and `scripts/train_dpo.sh`.
4. Run `bash scripts/train_sft.sh` or `bash scripts/train_dpo.sh`.

### How to evaluate a checkpoint

1. Prepare the dataset following `scripts/example_sft.json`.
2. Modify the parameters in `scripts/eval.sh`.
3. Run `bash scripts/eval.sh`.

## 👀 Team

**Team Tsinghua**: Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Chao Zhang

**Team ByteDance**: Wei Li, Zejun Ma

## ✨ Citation
If you find video-SALMONN 2 useful, please cite the paper:

```
@article{tang2025video,
    title={{video-SALMONN 2: Captioning-Enhanced Audio-Visual Large Language Models}}, 
    author={Changli Tang and Yixuan Li and Yudong Yang and Jimin Zhuang and Guangzhi Sun and Wei Li and Zejun Ma and Chao Zhang},
    journal={arXiv preprint arXiv:2506.15220},
    year={2025},
}
```
