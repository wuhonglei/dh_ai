from torch.nn.functional import mse_loss


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def compute_content_loss(gen_features, content_features):
    return mse_loss(gen_features.relu2_2, content_features.relu2_2)


def compute_style_loss(gen_features, style_grams):
    style_loss = 0.0
    for gen, style_gram in zip(gen_features, style_grams):
        style_loss += mse_loss(gram_matrix(gen),
                               style_gram.repeat(gen.size(0), 1, 1))
    return style_loss
