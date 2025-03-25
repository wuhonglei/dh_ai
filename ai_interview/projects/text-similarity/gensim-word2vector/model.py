from cv2 import accumulate
from gensim.models import Word2Vec
from git import Union
from typing import TypedDict
from utils.common import timer_decorator
from gensim.models.callbacks import CallbackAny2Vec
import wandb
from tqdm import tqdm
from cbow_dataset import CBOWDataset
import numpy as np


class CBOWModelConfig(TypedDict):
    sentences: Union[CBOWDataset, list[list[str]]]
    sg: int
    embedding_dim: int
    window: int
    min_count: int
    workers: int
    epochs: int


class ModelConfig(TypedDict):
    embedding_dim: int
    window: int
    min_count: int
    workers: int
    sg: int
    vocab_size: int


class EpochLogger(CallbackAny2Vec):
    def __init__(self, epochs: int, total_samples: int):
        self.epoch = 0
        self.total_loss = 0
        self.total_samples = total_samples
        self.epochs = epochs
        self.epoch_bar = tqdm(range(epochs), desc="Epoch")

    def on_epoch_end(self, model):
        accumulate_loss = model.get_latest_training_loss()
        current_loss = accumulate_loss - self.total_loss
        self.total_loss = accumulate_loss
        avg_loss = current_loss / self.total_samples  # 计算平均损失
        self.epoch_bar.set_postfix(avg_loss=avg_loss)
        # 可以添加 wandb 记录
        wandb.log({
            'epoch': self.epoch,
            'avg_loss': avg_loss
        })
        self.epoch += 1
        self.epoch_bar.update(1)


class CBOWModel:
    @timer_decorator
    def __init__(self, config: CBOWModelConfig | None = None):
        self.model: Word2Vec | None = None
        self.config = config or {}

    @timer_decorator
    def train(self):
        self.model = Word2Vec(sentences=self.config['sentences'],
                              sg=self.config['sg'],
                              window=self.config['window'],
                              min_count=self.config['min_count'],
                              vector_size=self.config['embedding_dim'],
                              workers=self.config['workers'],
                              epochs=self.config['epochs'],
                              compute_loss=True,
                              callbacks=[EpochLogger(
                                  self.config['epochs'], len(self.config['sentences']))]
                              )

    def get_config(self) -> ModelConfig:
        """获取模型配置"""
        if self.model is None:
            raise ValueError("Model not trained")

        return {
            'embedding_dim': self.model.vector_size,
            'window': self.model.window,
            'min_count': self.model.min_count,
            'sg': self.model.sg,
            'workers': self.model.workers,
            'vocab_size': len(self.model.wv.key_to_index)
        }

    def get_word_vector(self, word: str):
        if self.model is None:
            raise ValueError("Model not trained")

        if word not in self.model.wv:
            return np.zeros(self.model.vector_size)

        return self.model.wv[word]

    def encode(self, words: list[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        word_vectors = np.array([self.get_word_vector(word) for word in words])
        return word_vectors.mean(axis=0)

    @timer_decorator
    def save(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained")
        config = self.get_config()
        wandb.log({'model_config': config})  # 记录到 wandb
        self.model.save(path)

    @timer_decorator
    def load(self, path: str):
        self.model = Word2Vec.load(path)


if __name__ == "__main__":
    sentences = [["cat", "say", "meow"], ["dog", "say", "woof"]]
    model = CBOWModel(config={
        'sentences': sentences,
        'sg': 1,
        'embedding_dim': 100,
        'window': 5,
        'min_count': 1,
        'workers': 4,
        'epochs': 5
    })
    print(model.get_word_vector("cat"))
