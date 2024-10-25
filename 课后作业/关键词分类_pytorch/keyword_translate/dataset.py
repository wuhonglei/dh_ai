from torch.utils.data import DataLoader, Dataset


class KeywordDataset(Dataset):
    def __init__(self, keywords):
        self.keywords = keywords

    def __len__(self):
        return len(self.keywords)

    def __getitem__(self, idx):
        return self.keywords[idx], idx
