'''
    code by TaeHwan Jung(@graykode)
    Original Paper and repository here : https://github.com/openai/gpt-2
    GPT2 Pytorch Model : https://github.com/huggingface/pytorch-pretrained-BERT
'''
import logging
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)


def load_weight(model, state_dict):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if key.endswith(".g"):
            new_key = key[:-2] + ".weight"
        elif key.endswith(".b"):
            new_key = key[:-2] + ".bias"
        elif key.endswith(".w"):
            new_key = key[:-2] + ".weight"
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_model = model
    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    load(start_model, prefix="")

    # Make sure we are still sharing the output and input embeddings after loading weights
    model.set_tied()
    return model


# 比较两个模型的权重
def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.state_dict().items(), model2.state_dict().items()):
        if name1 != name2:
            print(f"参数名称不同: {name1} vs {name2}")
            return False
        if not torch.equal(param1, param2):
            print(f"参数值不同: {name1}")
            return False
    return True


def get_device():
    return 'cpu'

    if torch.backends.mps.is_available():  # type: ignore
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def compute_loss(lm_logits, lm_labels):
    # 1. 检查输入维度
    expected_shape = lm_logits.size()[:-1]  # 除最后一维外应该相同
    if lm_labels.size() != expected_shape:
        raise ValueError(
            f"标签形状 {lm_labels.size()} 与预期形状 {expected_shape} 不匹配"
        )

    # 2. 确保数据类型正确
    if not lm_labels.dtype == torch.long:
        lm_labels = lm_labels.long()

    # 3. 创建损失函数
    loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    # 4. 重塑张量并计算损失
    loss = loss_fct(
        # [batch_size * seq_len, vocab_size]
        lm_logits.view(-1, lm_logits.size(-1)),
        lm_labels.view(-1)                        # [batch_size * seq_len]
    )

    return loss
