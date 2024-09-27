import torch
import torch.nn as nn


class SkipGramModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(SkipGramModel, self).__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_size)
        self.out_embed = nn.Embedding(vocab_size, embed_size)

    def forward(self, center, context, negative):
        center_embeds = self.in_embed(center)  # (batch_size, embedding_dim)
        context_embeds = self.out_embed(context)  # (batch_size, embedding_dim)
        # (batch_size, negative_samples, embedding_dim)
        negative_embeds = self.out_embed(negative)

        pos_score = torch.sum(torch.mul(center_embeds, context_embeds), dim=1)
        pos_score = torch.sigmoid(pos_score)

        if negative.size(0) != 0:
            neg_score = torch.bmm(
                negative_embeds, center_embeds.unsqueeze(2)).squeeze()
            neg_score = torch.sum(torch.sigmoid(-neg_score), dim=1)
        else:
            neg_score = torch.FloatTensor([1] * center.size(0))

        return -1 * (torch.log(pos_score) + torch.log(neg_score)).mean()
