from torch.utils.data import Dataset


class WritingPromptsDataset(Dataset):
    def __init__(self, prompt_path: str, story_path: str):
        self.prompt_file = self._load_file(prompt_path)
        self.story_file = self._load_file(story_path)

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
