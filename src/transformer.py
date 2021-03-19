import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    """
    Implement the transformer network
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)



class Generator(nn.Module):
    """
    Project from a d_model dim vector to a scalar(final estimation)
    """
    def __init__(self, d_model):
        super(Generator, self).__init__()
        self.proj = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.ReLU(),
                                  nn.Linear(d_model, 1))

    def forward(self, x):
        return self.proj(x)


def clones(module, N):
    """
    Produce N identical layers
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    """
    implement layernorm algorithm
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    Glue layer, connnect a sublayer with a residual connection
    and normalization
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class Embedding(nn.Module):
    """
    Project the d_feature dim vector to d_model dim vector
    """
    def __init__(self, d_feature, d_model):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.lut = nn.Sequential(nn.Linear(d_feature, d_model),
                                 nn.ReLU(),
                                 nn.Linear(d_model, d_model))

    def forward(self, x):
        return self.lut(x)




class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """
    def __init__(self, layer, N):
        super(Encoder,self).__init__()
        # self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        # for layer in self.layers:
        #     x = layer(x,mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Consists of a self attention layer and a feedforward layer
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    """
    Core decoder is a stack of N layers
    """
    def __init__(self, layer, N ):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Consists of one self attention layer, one encoder-decoder attention layer,
    and one feedforward layer
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(mask):
    """
    mask future obs
    """
    device = mask.device
    size = mask.size(-2)
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    subsequent_mask =  torch.from_numpy(subsequent_mask) == 0
    subsequent_mask = subsequent_mask.to(device)
    return subsequent_mask


def missing_mask(mask):
    """
    mask obs based on mask
    """
    # mask has dimension (batch_size, seq_len, 1)
    device = mask.device
    mask = mask.squeeze(2)
    size = mask.size(1)
    missing_mask = torch.ones((size, size), device=device) * mask
    off_diagonal_mask = torch.diag(torch.ones(size - 1), diagonal=1)
    off_diagonal_mask = off_diagonal_mask.to(device)
    missing_mask.masked_fill_(off_diagonal_mask == 1, 0)
    return missing_mask.unsqueeze(0) == 1

def make_std_mask(mask, see_future):
    """
    Combine missing_mask and subsequent_mask together
    """
    if not see_future:
        total_mask =  subsequent_mask(mask) & missing_mask(mask)
    else:
        total_mask = missing_mask(mask)

    return total_mask.type_as(mask.data)





def attention(query, key, value, mask=None, dropout=None):
    """
    Core attention function
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Sublayer: multi-head attention
    """
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask = None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model to h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l,x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """
    Sublayer: Feedforward net
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))





class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)




def make_model(d_feature, N = 1, d_model=6, d_ff=2, h=2, dropout=0.1):
    """
    assemble the transformer neural net
    """
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout),N),
        nn.Sequential(c(Embedding(d_feature, d_model)), c(position)),
        nn.Sequential(c(Embedding(1, d_model)), c(position)),
        Generator(d_model)
    )

    return model


if __name__ == "__main__":
    model = make_model(2)
    model.train()
    src = torch.rand((1,2,2))
    trt = torch.rand((1,2,1))
    trt_mask = torch.tensor([1,1]).view(1,-1,1)
    #trt_mask = missing_mask(trt_mask)
    #trt_mask = subsequent_mask(trt_mask)
    trt_mask = make_std_mask(trt_mask)

    print(trt_mask)
    src_mask = None
    print(model(src, trt, src_mask, trt_mask))

