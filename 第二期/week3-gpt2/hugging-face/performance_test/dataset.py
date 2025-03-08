from random import random
from torch.utils.data import Dataset, Sampler


class WritingPromptsDataset(Dataset):
    def __init__(self, prompt_path: str, story_path: str):
        self.prompt_file = self._load_file(prompt_path)[:1000]
        self.story_file = self._load_file(story_path)[:1000]

    def __len__(self):
        return len(self.prompt_file)

    def _load_file(self, file_path: str):
        with open(file_path, "r") as f:
            return [line.strip() for line in f.readlines()]

    def __getitem__(self, idx):
        prompt = self.prompt_file[idx]
        story = self.story_file[idx]
        return {
            'prompt': prompt,
            'story': story,
        }


class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, sort_key, bucket_size=100):
        self.dataset = dataset
        self.batch_size = batch_size
        self.bucket_size = bucket_size

        # 计算数据长度并排序
        lengths = [(i, sort_key(dataset[i])) for i in range(len(dataset))]

        # 分桶并在每个桶内排序
        self.buckets = []
        for i in range(0, len(lengths), bucket_size):
            bucket = lengths[i:i + bucket_size]
            bucket.sort(key=lambda x: x[1])  # 按长度排序
            self.buckets.append([x[0] for x in bucket])

    def __iter__(self):
        # 随机打乱桶的顺序
        from random import shuffle
        shuffle(self.buckets)
        indices = []
        for bucket in self.buckets:
            indices.extend(bucket)
        return iter(indices)

    def __len__(self):
        return len(self.dataset)
