import math
from dataset_metadata.data_field_name import DataFieldName
import torch
from typing import Dict
from .rope2d import get_rope_index_25
from transformers import ProcessorMixin, AutoTokenizer, Qwen2VLImageProcessorFast, WhisperFeatureExtractor
import torchaudio
from ltx_video.utils.mel_spectrogram import to_mono

def split_into_groups(count, group):
    base = count // group
    remainder = count % group
    return [base + 1] * remainder + [base] * (group - remainder)

def resample_and_downmix(audio, audio_sample_rate, target_sr):
    monos = []

    for audio, sr in zip(audio, audio_sample_rate):
        monos.append(torchaudio.functional.resample(to_mono(audio), orig_freq=sr, new_freq=target_sr))

    return monos

def generate_id_target(self, conversation, thw, audio_lengths):
    IGNORE_INDEX = -100
    T,H,W = thw

    input_id, target = [], []

    for message in conversation:
        role = message["role"]
        content = message["content"]

        if role == "user":
            content +=  "<|vision_start|>"
            if audio_lengths is None:
                content += (
                    + f"<|video_pad|>"
                    * (T*H*W // self.image_processor.merge_size**2)
                )
            else:
                per_timestep_audio_len = split_into_groups(audio_lengths, T)
                for timestep in range(T):
                    content += (
                        f"<|video_pad|>" 
                        * (H*W // self.image_processor.merge_size**2)
                        + f"<|audio_pad|>"
                        * per_timestep_audio_len[timestep]
                    )

            content += "<|vision_end|>"

        encode_id = self.tokenizer.apply_chat_template([{"role": role, "content": content}])
        input_id += encode_id
        
        if role == "user":    
            target += [IGNORE_INDEX] * len(encode_id)
        else:
            PREFIX_TOKENS = 2 # <|im_start|>assistant
            SUFFIX_TOKENS = 2 # <|im_end|>\n
            
            target_mask = encode_id.copy()            
            target_mask[:PREFIX_TOKENS] = [IGNORE_INDEX] * PREFIX_TOKENS
            target_mask[-SUFFIX_TOKENS:] = [IGNORE_INDEX] * SUFFIX_TOKENS

            target += target_mask

    return input_id, target

class SalmonnProcessor(ProcessorMixin):

    def __init__(self, model_args, training_args, data_args, sampling_rate=16000):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        self.image_processor = Qwen2VLImageProcessorFast.from_pretrained(
            model_args.model_base
        )

        self.audio_processor = WhisperFeatureExtractor(
            feature_size=data_args.feature_size, # to do in model args
            sampling_rate=data_args.sampling_rate,
            hop_length=data_args.hop_length,
            chunk_length=data_args.chunk_length,
        )

        self.sampling_rate = sampling_rate
        
    def preprocess_qwen_2_visual(self, batch) -> Dict:
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        self.tokenizer.chat_template = chat_template

        input_ids, targets = [], []

        for conversation, thw, audio_lengths in zip(batch['conversations'], batch['grid_thw'], batch['audio_lengths']):
            input_id, target = generate_id_target(conversation, thw, audio_lengths)
            assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            input_ids.append(input_id)
            targets.append(target)

        batch['input_ids'] = torch.tensor(input_ids, dtype=torch.long)
        batch['targets'] = torch.tensor(targets, dtype=torch.long)

        return batch

    def process_video_frames(self, batch):
        for subsampled_video in batch['subsampled_video']:
            video_length = len(batch['frames']) / batch['average_frame_rate']
            fps = len(subsampled_video) / video_length

            self.image_processor.max_pixels = self.data_args.video_max_frame_pixels * max(self.video_max_frames / len(subsampled_video))
            self.image_processor.min_pixels = self.data_args.video_min_frame_pixels
            self.image_processor.size["longest_edge"] = self.image_processor.max_pixels
            self.image_processor.size["shortest_edge"] = self.image_processor.min_pixels
            video_processed = self.image_processor.preprocess(
                images=None, videos=[subsampled_video], return_tensors="pt"
            )

            batch['video_tensor'].extend(video_processed["pixel_values_videos"])
            batch['grid_thw'].extend(video_processed["video_grid_thw"][0])
            batch['grid_temporal_resolution'].append(self.processor.temporal_patch_size / fps)

        return batch

    def process_audio(self, batch):
        THIRTY_SECONDS = 30 * self.sampling_rate

        batch['audio_inputs'] = []
        batch['audio_lengths'] = []

        for audio_data in resample_and_downmix(batch['audio'], batch['audio_sample_rate']):
            steps = audio_data.shape[0]
            if audio_data.shape[0] < self.sampling_rate:
                padding = self.sampling_rate - steps
                audio_data = torch.pad(audio_data, (0, padding), mode="constant", constant_values=0)

            chunks = torch.split(audio_data, THIRTY_SECONDS)
            
            spectrogram_lst = [self.audio_processor(a, sampling_rate=self.sampling_rate, return_tensors="pt")["input_features"].squeeze() for a in chunks]
            batch['audio_feature'].append(torch.stack(spectrogram_lst, dim=0))
            batch['audio_lengths'].append(math.ceil(len(audio_data) / THIRTY_SECONDS) * 60)
        
        return batch

    def get_rope_index_25(self, batch):
        batch['position_ids'] = get_rope_index_25(
            batch["input_ids"],
            image_grid_thw=torch.stack(batch['grid_thw'], dim=0),
            video_grid_thw=torch.stack(batch['video_grid_thw'], dim=0),
            second_per_grid_ts=batch['second_per_grid_ts'],
            audio_lengths=batch['audio_lengths'],
            merge_size=self.image_processor.merge_size,
        )[0]

        return batch

    def batch_decode(self, batch):
        # user_prompt = "Please provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so."
        batch = self.process_video_frames(batch)
        batch = self.process_audio(batch)
        
        batch['conversations'] = [
            [
                {"from": "user", "value": f"<video>"},
                {"from": "assistant",   "value": f"{caption}"},
            ] 
            for caption in batch[DataFieldName.AUDIO_VISUAL_CAPTION] 
        ]

        batch = self.preprocess_qwen_2_visual(batch)
        batch = self.get_rope_index_25(batch)
        batch['train_type'] = 'sft'

        return batch
