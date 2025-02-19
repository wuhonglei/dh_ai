wandb_config = {
    'project': 'CLIP',
    'config': {
        'data': {
            'train_data_path': "./datasets/train/captions.csv",
            'test_data_path': "./datasets/test/captions.csv",
            'img_path': "./datasets/images"
        },
        'text_encoder': {
            'model_name': "distilbert-base-uncased",
            'embedding_dim': 768,
            'pretrained': True,
            'trainable': False,
            'max_length': 200
        },
        'image_encoder': {
            'model_name': "resnet50",
            'input_size': 224,
            'embedding_dim': 2048,
            'pretrained': True,
            'trainable': False
        },
        'projection_head': {
            'embedding_dim': 256,
            'dropout': 0.1
        },
        'train': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 3,
            'shuffle': True,
            'temperature': 1.0,
        },
        'shutdown': False,
        'sweep': False,
    },
    'job_type': 'train',
}

sweep_config = {
    'method': 'grid',  # 调参方法：random / grid / bayes
    'program': 'train.py',
    'metric': {
        'name': 'test_accuracy',  # 优化目标
        'goal': 'maximize'  # 最大化验证集准确率
    },
    'parameters': {
        'n_heads': {
            'values': [8, 12]
        },
        'depth': {
            'values': [8, 12]
        },
    }
}
