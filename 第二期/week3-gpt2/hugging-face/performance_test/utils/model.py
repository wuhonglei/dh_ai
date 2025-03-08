import torch
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedModel


def create_optimized_model(model_name, device) -> PreTrainedModel:
    # 1. 配置模型
    config = GPT2Config.from_pretrained(
        model_name,
        use_flash_attention=True,  # 启用flash attention
        use_memory_efficient_attention=True  # 使用内存效率更高的attention
    )

    # 2. 加载模型
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,  # 使用半精度
        device_map="auto"  # 自动处理设备映射
    )

    # 3. 模型优化
    model = model.to(device)

    # 4. 编译模型（如果可用）
    if torch.__version__ >= "2.0.0":
        try:
            model = torch.compile(
                model,
                mode='reduce-overhead',
                fullgraph=True
            )
        except Exception as e:
            print(f"Model compilation failed: {e}")

    return model  # type: ignore
