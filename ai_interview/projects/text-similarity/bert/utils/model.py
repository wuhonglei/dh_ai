from type_definitions import WandbConfig


def get_model_final_name(config: WandbConfig) -> str:
    batch_size = config.batch_size
    epochs = config.epochs
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    use_projection = config.use_projection

    return f"./cache/batch_size{batch_size}-epochs{epochs}-lr{learning_rate}-wd{weight_decay}-projection{use_projection}_final.pth"
