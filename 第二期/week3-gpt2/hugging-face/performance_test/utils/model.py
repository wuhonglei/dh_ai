from typing import List
import torch
from transformers import GPT2LMHeadModel, GPT2Config, PreTrainedModel, PreTrainedTokenizer


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
    )

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


# 性能优化设置
def setup_performance_opts():
    if torch.cuda.is_available():
        # 启用 TF32
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
        # 启用 cuDNN 基准测试
        # torch.backends.cudnn.benchmark = True  # type: ignore
        # 可选：设置确定性计算
        # torch.backends.cudnn.deterministic = True


def generate_text(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prompt_list: List[str], max_length: int, device) -> List[str]:
    try:
        inputs = tokenizer(
            prompt_list, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,  # type: ignore
                do_sample=False,
                use_cache=True,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        generated_list = tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        # 移除提示部分，仅保留生成内容
        return [generated.replace(prompt, "").strip() for generated, prompt in zip(generated_list, prompt_list)]
    except Exception as e:
        print(f"Error: {e}")
        return [''] * len(prompt_list)
