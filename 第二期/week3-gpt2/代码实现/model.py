# GPT-2 代码实现

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from config import GPT2Config


def gelu(x):
    # 添加 gelu 激活函数定义
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Conv1D(nn.Module):
    def __init__(self, nx: int, nf: int):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


class Attention(nn.Module):
    def __init__(self, nx: int, n_ctx: int, config: GPT2Config, scale: bool = False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        n_ctx = n_ctx
        assert n_state % config.n_head == 0, "Input embedding dimension must match n_embd"
        self.d_k = n_state // config.n_head
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(nx, nx * 3)
        self.c_proj = Conv1D(nx, nx)
        self.register_buffer('bias', torch.tril(  # 下三角矩阵, 右上角元素为 0
            torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def _attn(self, q, k, v):
        # q, k, v 形状: [batch_size, seq_len, nx]
        batch_size, seq_len, nx = q.size()
        assert nx % self.n_head == 0, "Input embedding dimension must match n_embd"

        # 将张量分割为多头形式
        # 新形状: [batch_size, seq_len, n_head, nx/n_head]
        q = q.view(batch_size, q.size(1), self.n_head, self.d_k)
        k = k.view(batch_size, k.size(1), self.n_head, self.d_k)
        v = v.view(batch_size, v.size(1), self.n_head, self.d_k)

        # 转置以便进行批量矩阵乘法
        # 新形状: [batch_size, n_head, seq_len, nx/n_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_mask = self.bias[:, :, :seq_len, :seq_len]  # type: ignore
        attn_weights = attn_weights.masked_fill(attn_mask == 0, -1e9)

        # 缩放注意力分数
        if self.scale:
            attn_weights = attn_weights / math.sqrt(self.d_k)

        # 应用softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 计算输出
        out = torch.matmul(attn_weights, v)

        # 转置回原始形状
        # 新形状: [batch_size, seq_len, n_head, nx/n_head]
        out = out.transpose(1, 2).contiguous()

        # 合并多头
        # 最终形状: [batch_size, seq_len, nx]
        out = out.view(batch_size, seq_len, nx)

        return out

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        # query, key, value 形状: [batch_size, seq_len, n_embd]
        query, key, value = x.split(self.split_size, dim=-1)
        if layer_past is not None:
            key = torch.cat((layer_past[0], key), dim=-2)
            value = torch.cat((layer_past[1], value), dim=-2)

        present = torch.stack((key, value))
        a = self._attn(query, key, value)
        a = self.c_proj(a)
        return a, present


class MLP(nn.Module):
    def __init__(self, n_state: int, config: GPT2Config):
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(nx, n_state)
        self.c_proj = Conv1D(n_state, nx)
        self.act = gelu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present


class GPT2Model(nn.Module):
    def __init__(self, config):
        super(GPT2Model, self).__init__()
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block)
                               for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)  # 旧的序列长度
        position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                    device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        x = inputs_embeds + position_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            x, present = block(x, layer_past=layer_past)
            presents.append(present)
        x = self.ln_f(x)
        return x, presents


class GPT2LMHead(nn.Module):
    def __init__(self, model_embeddings_weights, config):
        super(GPT2LMHead, self).__init__()
        self.n_embd = config.n_embd
        self.decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.set_embeddings_weights(model_embeddings_weights)

    def set_embeddings_weights(self, model_embeddings_weights):
        self.decoder.weight = model_embeddings_weights

    def forward(self, hidden_state):
        # h_trunc = hidden_state[:, :-1].contiguous().view(-1, self.n_embd)
        lm_logits = self.decoder(hidden_state)
        return lm_logits


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(config)
        self.lm_head = GPT2LMHead(self.transformer.wte.weight, config)

    def forward(self, input_ids, lm_labels=None, past=None):
        hidden_states, presents = self.transformer(input_ids, past=past)
        lm_logits = self.lm_head(hidden_states)
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            return loss
        return lm_logits, presents

    def set_tied(self):
        self.lm_head.set_embeddings_weights(self.transformer.wte.weight)
