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
            neg_score = torch.sigmoid(-neg_score)
        else:
            neg_score = torch.FloatTensor([1] * center.size(0))

        return -1 * (torch.log(pos_score) + torch.sum(torch.log(neg_score), dim=1)).mean()

    def similar_word(self, word, word2idx, idx2word, top_n=10):
        word_idx = word2idx[word]
        word_embed = self.in_embed(torch.LongTensor([word_idx]))
        word_embed = word_embed.repeat(len(word2idx), 1)
        embeddings = self.in_embed(
            torch.LongTensor(list(range(len(word2idx)))))
        cosine_sim = nn.CosineSimilarity(dim=1)(word_embed, embeddings)
        sim, indices = torch.topk(cosine_sim, top_n + 1)
        result = [(idx2word[idx.item()], sim[i].item())
                  for i, idx in enumerate(indices)][1:]
        return result
