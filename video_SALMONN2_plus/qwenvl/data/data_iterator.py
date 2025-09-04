import io
from dataset_metadata.data_field_name import DataFieldName
from txt2img.common import dist_util
from txt2img.common.machine import AccelerationType
from txt2img.config.data import DatasetConfig, DatasetFieldConfig
from txt2img.data_loaders.native_dataloader.native_dataloader_factory import DataloaderFactory

def wrap_dataloader_as_qwen_sources(dataloader):
    while True:
        batch = next(dataloader)  

        videos = batch[DataFieldName.VIDEO_ORIGINAL]
        captions = batch.get(DataFieldName.AUDIO_VISUAL_CAPTION, [None] * len(videos))

        sample  = {
            "video": [io.BytesIO(v) for v in videos],
            "conversations": [
                {"from": "human", "value": f"<video>\nPlease provide a thorough description of all the content in the video, including every detail. As you describe, ensure that you also cover as much information from the audio as possible, and be mindful of the synchronization between the audio and video as you do so."},
                {"from": "gpt",   "value": "\n".join([str(c) or "" for c in captions])},
            ],
            "should_use": True,
            "use_audio": True,
        }

        yield sample


def make_qwen_lazy_iterator(batch_size: int, *, debug_mode: bool = False, loop: bool = False):
    try:
        dist_util.init(acceleration_type=AccelerationType.GPU, devices_per_vm=1)
    except:
        pass
    
    dataset_config = DatasetConfig(
        dataset_metadata=["mevaseret-v2"],
        region="europe-west4",
        fields=[
            DatasetFieldConfig(name=DataFieldName.VIDEO_ORIGINAL),
            DatasetFieldConfig(name=DataFieldName.AUDIO_VISUAL_CAPTION),
        ],
        mappers=[],
        debug_mode=debug_mode,
    )

    dataloader = DataloaderFactory.create_data_loader(
        dataset_config,
        loop=loop,
        batch_size=batch_size,
    )

    dataloader = dist_util.BackgroundDeviceLoader(
        dataloader=dataloader,
        pl_kwargs={
            "batches_per_execution": 1,
            "loader_prefetch_size": 1,
            "device_prefetch_size": 1,
            "host_to_device_transfer_threads": 1,
        },
    )

    iterator = iter(dataloader)
    return wrap_dataloader_as_qwen_sources(iterator)