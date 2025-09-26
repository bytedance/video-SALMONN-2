# video-SALMONN 2: Caption-Enhanced Audio-Visual Large Language Models

🚀🚀 Welcome to the repo of **video-SALMONN 2**!

video-SALMONN 2 is a powerful audio-visual large language model (LLM) that **generates high-quality audio-visual video captions**, which is developed by the Department of Electronic Engineering at Tsinghua University and ByteDance. 

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2506.15220'><img src='https://img.shields.io/badge/video_SALMONN_2_paper-PDF-green'></a>
<a href='https://video-salmonn-2.github.io'><img src='https://img.shields.io/badge/Demo-link-green'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2'><img src='https://img.shields.io/badge/video_SALMONN_2_7B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_3B'><img src='https://img.shields.io/badge/video_SALMONN_2+_3B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B'><img src='https://img.shields.io/badge/video_SALMONN_2+_7B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_72B'><img src='https://img.shields.io/badge/video_SALMONN_2+_72B-checkpoint-yellow'></a>
<a href='https://huggingface.co/datasets/tsinghua-ee/video-SALMONN_2_testset'><img src='https://img.shields.io/badge/video_SALMONN_2-testset-yellow'></a>
</div>

## 🔥 News

- **2025-09-26**: A new version (Version-2509) of video-SALMONN 2+ is released, containing minor [code](https://github.com/bytedance/video-SALMONN-2/tree/main/video_SALMONN2_plus) revision, an update for [7B model](https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B) and [72B model](https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_72B), as well as the addition of [3B model](https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_3B). The upgraded video-SALMONN 2+ further enhances audio-visual and visual-only understanding capability on various benchmarks. 
- **2025-07-17**: We release the code and checkpoint of video-SALMONN 2+ at [video-SALMONN 2+](https://github.com/bytedance/video-SALMONN-2/tree/main/video_SALMONN2_plus) （Version-2507). video-SALMONN 2+ achieves SOTA results on [Video-MME](https://video-mme.github.io/home_page.html) benchmark.
- **2025-07-08**: We release the 7B version of video-SALMONN 2.
- **2025-06-18**: We release the code of video-SALMONN 2.

## ⚡️ Results
We evaluate the models on audio-visual QA benchmarks including Video-MME, WorldSense, AVUT, Video-Holmes, and DailyOmni, and visual-only benchmarks including MLVU and LVBench. Our 3B and 7B models achieve SOTA results at comparable scales, while the 72B model surpasses all other open-source systems.

<img width="857" height="510" alt="image" src="https://github.com/user-attachments/assets/aca20b2e-1e68-4b44-a26b-03d5f070b213" />


## 🌈 How to Use

### How to train video-SALMONN 2

1. Prepare the dataset following `scripts/example_sft.json` and `scripts/example_dpo.json`.
2. Download LLaVA-OneVision Model from [huggingface](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov).
3. Modify the parameters in `scripts/train_sft.sh` and `scripts/train_dpo.sh`.
4. Run `bash scripts/train_sft.sh` or `bash scripts/train_dpo.sh`.

### How to evaluate a checkpoint

1. Prepare the dataset following `scripts/example_sft.json`.
2. Modify the parameters in `scripts/eval.sh`.
3. Run `bash scripts/eval.sh`.

### For video-SALMONN 2+, please refer to [video_SALMONN2_plus](https://github.com/bytedance/video-SALMONN-2/tree/main/video_SALMONN2_plus)

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
