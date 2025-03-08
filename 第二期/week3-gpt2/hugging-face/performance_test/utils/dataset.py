from typing import Literal, List, Tuple
from torch.utils.data import DataLoader
from dataset import WritingPromptsDataset
from torch.utils.data import DistributedSampler


def get_dataloaders(splits: List[Literal['train', 'test', 'val']], batch_size: int, is_distributed: bool) -> Tuple[DataLoader]:
    path_dict = {
        'train': {
            'prompt': 'writingPrompts/train.wp_source',
            'story': 'writingPrompts/train.wp_target',
        },
        'test': {
            'prompt': 'writingPrompts/test.wp_source',
            'story': 'writingPrompts/test.wp_target',
        },
        'val': {
            'prompt': 'writingPrompts/valid.wp_source',
            'story': 'writingPrompts/valid.wp_target',
        },
    }

    data_loaders = []
    for split in splits:
        dataset = WritingPromptsDataset(  # type: ignore
            prompt_path=path_dict[split]['prompt'],
            story_path=path_dict[split]['story'],
        )

        if is_distributed:
            # 分布式环境下使用 DistributedSampler，推理时不需要 shuffle
            sampler = DistributedSampler(
                dataset,
                shuffle=False,
                drop_last=False  # 推理时不丢弃最后的不完整批次
            )
        else:
            # 非分布式环境下使用 BucketSampler
            sampler = None

        data_loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=False  # 推理时不丢弃最后的不完整批次
            )
        )

    return tuple(data_loaders)
