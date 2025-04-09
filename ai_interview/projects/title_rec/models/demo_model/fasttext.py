import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class FastText(nn.Module):
    """FastText model implementation using PyTorch.

    This implementation follows the FastText architecture with subword embeddings
    and a linear classifier.

    Attributes:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of word embeddings
        num_classes (int): Number of output classes
        min_n (int): Minimum length of subwords
        max_n (int): Maximum length of subwords
        word_embeddings (nn.Embedding): Embedding layer for words
        classifier (nn.Linear): Linear classifier layer
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 100,
        num_classes: int = 30,
        min_n: int = 2,
        max_n: int = 3,
        padding_idx: Optional[int] = None
    ):
        """Initialize the FastText model.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            num_classes: Number of output classes
            min_n: Minimum length of subwords
            max_n: Maximum length of subwords
            padding_idx: Index for padding token
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.min_n = min_n
        self.max_n = max_n

        # Initialize embedding layer
        self.word_embeddings = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # Initialize classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        nn.init.normal_(self.word_embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def get_subwords(self, word: str) -> List[str]:
        """Generate subwords for a given word.

        Args:
            word: Input word

        Returns:
            List of subwords including the full word
        """
        word = f"<{word}>"  # Add boundary symbols
        subwords = [word]   # Include full word
        for n in range(self.min_n, self.max_n + 1):
            for i in range(len(word) - n + 1):
                subwords.append(word[i:i+n])
        return subwords

    def get_text_vector(self, text: str, word_to_idx: Dict[str, int]) -> torch.Tensor:
        """Convert text to vector representation using average pooling.

        Args:
            text: Input text
            word_to_idx: Dictionary mapping words to indices

        Returns:
            Text vector representation
        """
        words = text.split()  # Assume text is already tokenized
        if not words:
            return torch.zeros(self.embedding_dim)

        vectors = []
        for word in words:
            subwords = self.get_subwords(word)
            for subword in subwords:
                if subword in word_to_idx:
                    idx = word_to_idx[subword]
                    vectors.append(self.word_embeddings(torch.tensor(idx)))

        if not vectors:
            return torch.zeros(self.embedding_dim)

        vectors = torch.stack(vectors)
        return torch.mean(vectors, dim=0)

    def forward(self, text: str, word_to_idx: Dict[str, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            text: Input text
            word_to_idx: Dictionary mapping words to indices

        Returns:
            Tuple of (predicted class, probability distribution)
        """
        text_vector = self.get_text_vector(text, word_to_idx)
        logits = self.classifier(text_vector)
        probs = F.softmax(logits, dim=0)
        return torch.argmax(probs), probs


# Example usage
if __name__ == "__main__":
    # Create vocabulary mapping
    word_to_idx = {
        "苹果": 0,
        "手机壳": 1,
        "红富士": 2,
        "华为": 3,
        "<苹果>": 4,
        "<手机壳>": 5,
        "<红富士>": 6,
        "<华为>": 7,
        # Add more words and subwords as needed
    }

    # Initialize model
    model = FastText(
        vocab_size=len(word_to_idx),
        embedding_dim=100,
        num_classes=2,
        min_n=2,
        max_n=3
    )

    # Example training data
    data = [("苹果 手机壳", 0), ("红富士 苹果", 1)]  # 类别: 0=手机配件, 1=水果

    # Training loop
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    for text, label in data:
        for _ in range(10):  # Simulate multiple iterations
            optimizer.zero_grad()
            text_vector = model.get_text_vector(text, word_to_idx)
            logits = model.classifier(text_vector)
            loss = criterion(logits.unsqueeze(0), torch.tensor([label]))
            loss.backward()
            optimizer.step()

    # Prediction
    test_text = "华为 手机壳"
    pred_label, probs = model(test_text, word_to_idx)
    print(f"预测类别: {pred_label.item()}, 概率分布: {probs[:5]}...")
