from model import CBOWModel
from config import DATASET_CONFIG, VOCAB_CONFIG,   VocabConfig, CONFIG, WandbConfig
from utils.common import get_device
from utils.train import get_checkpoint_path_final, get_train_dataset_cache_path
from cbow_dataset import CBOWDataset
from vocab import Vocab
from dataset import NewsDatasetCsv
import os
import wandb


project = "text-similarity-gensim"


def train(_config: dict = {}):
    wandb.init(project=project, config={
        "toml": CONFIG.model_dump(), **_config
    })
    config: WandbConfig = wandb.config  # type: ignore
    min_freq = config.min_freq
    embedding_dim = config.embedding_dim
    epochs = config.epochs
    window = config.window
    num_workers = 4
    sg = config.sg

    vocab = Vocab(VocabConfig(
        **{**VOCAB_CONFIG.model_dump(), 'min_freq': min_freq}))

    train_csv_dataset = NewsDatasetCsv(DATASET_CONFIG.train_csv_dataset)
    train_dataset = CBOWDataset(train_csv_dataset, vocab)

    model = CBOWModel(config={
        'sentences': train_dataset,
        'sg': sg,
        'embedding_dim': embedding_dim,
        'window': window,
        'min_count': min_freq,
        'workers': num_workers,
        'epochs': epochs
    })
    model.train()
    # 保存最终模型并关闭 wandb
    model_file_path = get_checkpoint_path_final(config)
    wandb.run.summary["model_path"] = model_file_path  # type: ignore 作为摘要信息记录
    model.save(model_file_path)
    wandb.finish()


def main():
    use_sweep = True
    if not use_sweep:
        _config = {
            'min_freq': 5,
            'embedding_dim': 100,
            'epochs': 10,
            'window': 5,
            'sg': 0,
        }
        train(_config)
        return

    sweep_config = {
        'method': 'bayes',
        'metric': {'name': 'avg_loss', 'goal': 'minimize'},
        'parameters': {
            'min_freq': {'values': [200, 350, 500]},
            'embedding_dim': {'values': [100, 200, 300]},
            'epochs': {'values': [5, 10, 15]},
            'window': {'values': [2, 5, 8]},
            'sg': {'values': [0, 1]},
        }
    }
    use_exist_sweep = True
    if use_exist_sweep:
        os.environ['WANDB_PROJECT'] = project
        sweep_id = '6osyxyqb'
    else:
        sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, function=train, count=150)


if __name__ == "__main__":
    main()
