# video-SALMONN 2+: an updated version of video-SALMONN 2

video-SALMONN 2+ is built on Qwen 2.5-VL using a similar pipeline of video-SALMONN 2. Based on a better baseline and some other optimizations, video-SALMONN 2+ achieves SOTA on audio-visual QA benchmarks, including Video-MME, WorldSense, AVUT, Video-Holmes, and DailyOmni, and visual-only benchmarks including MLVU and LVBench. Our 3B and 7B models achieve SOTA results at comparable scales, while the 72B model surpasses all other open-source systems.

<div style='display:flex; gap: 0.25rem; '>
<a href='https://arxiv.org/abs/2506.15220'><img src='https://img.shields.io/badge/paper-PDF-green'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_7B'><img src='https://img.shields.io/badge/video_SALMONN_2+_7B-checkpoint-yellow'></a>
<a href='https://huggingface.co/tsinghua-ee/video-SALMONN-2_plus_72B'><img src='https://img.shields.io/badge/video_SALMONN_2+_72B-checkpoint-yellow'></a>
</div>

## Results
<img width="857" height="510" alt="image" src="https://github.com/user-attachments/assets/aca20b2e-1e68-4b44-a26b-03d5f070b213" />


## How to Use

**IMPORTANT**: To get the same evaluation result, please use `--max_frames 768 --max_pixels 61250`. Using excessively high resolution or frame rate for evaluation may lead to too much input token count for the model, potentially causing performance degradation.

1. Prepare the dataset following `scripts/example_av.json`, `scripts/example_v.json`, `scripts/example_dpo.json`, and `scripts/example_a.json`
2. Prepare base audio model through modifying the path in `gen_audio_model.py`
3. To conduct audio alignment, use the following script:
   ```bash
   bash scripts/train.sh --interval 0.1 --run_name audio_alignment --dataset path_to_dataset --lr 2e-5 --train_qformer --max_frames 768 --max_pixels 61250 --model path_to_audio_model --model_base path_to_audio_model --bs 16 --epoch 5 --save_steps 5000
   ```
4. To conduct audio-visual SFT, use the following script:
    ```bash
    bash scripts/train.sh --interval 0.1 --run_name av_sft --dataset path_to_dataset --lr 2e-5 --train_qformer --train_proj --max_frames 768 --max_pixels 61250 --model audio_align_model --model_base path_to_audio_model --epoch 5 --save_steps 2000 --use_lora --lora_r 128 --lora_alpha 256
    ```
5. To conduct DPO, use the following script:
    ```bash
    bash scripts/train.sh --interval 0.1 --run_name dpo --dataset path_to_dataset --max_frames 768 --max_pixels 61250 --model audio_visual_base --model_base audio_align_model --lora_ckpt audio_visual_checkpoint --train_type gdpo --use_lora --lora_r 128 --lora_alpha 256 --lr 5e-6 --epoch 1 --save_steps 200 --train_qformer --train_proj
    ```
6. To evaluate 7B model, use the following script:
   ```bash
   bash scripts/test.sh --interval 0.1 --run_name eval --dataset path_to_dataset --max_frames 768 --max_pixels 61250 --model path_to_audio_model --model_base path_to_audio_model --lora_ckpt model_ckpt
   ```
7. To evaluate 72B model, use the following script:
   ```bash
   bash scripts/test_8.sh --interval 0.1 --run_name eval --dataset path_to_dataset --max_frames 768 --max_pixels 61250 --model path_to_audio_model --model_base path_to_audio_model --lora_ckpt model_ckpt
   ```
