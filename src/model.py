import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        norm = self.weight * (x - mean) / (std + self.eps) + self.bias
        return norm
    
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.coeff = math.sqrt(d_model)
    
    def forward(self, x):
        return self.embedding(x) * self.coeff
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_weights = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, src_len, _ = key.size()

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.view(batch_size, 1, 1, src_len)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, self.attn_weights = attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        x = self.out_proj(x)

        return x

class EncoderLayer(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, dropout, norm_first=True):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_first = norm_first
    
    def forward(self, x, mask):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, mask))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x, mask):
        x = self.self_attn(x, x, x, mask)
        return self.dropout1(x)
    
    def _ff_block(self, x):
        x = self.feed_forward(x)
        return self.dropout2(x)
    
    
class Encoder(nn.Module):
    def __init__(self, layer, num_layers, norm):
        super(Encoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = norm
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        
        if self.norm is not None:
            x = self.norm(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, norm_first=True):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_first = norm_first
    
    def forward(self, x, memory, tgt_mask, src_mask):
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, src_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, src_mask))
            x = self.norm3(x + self._ff_block(x))
        return x
    
    def _sa_block(self, x, mask):
        x = self.self_attn(x, x, x, mask)
        return self.dropout1(x)
    
    def _mha_block(self, x, memory, mask):
        x = self.cross_attn(x, memory, memory, mask)
        return self.dropout2(x)
    
    def _ff_block(self, x):
        x = self.feed_forward(x)
        return self.dropout3(x)

class Decoder(nn.Module):
    def __init__(self, layer, num_layers, norm):
        super(Decoder, self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = norm
    
    def forward(self, x, memory, tgt_mask, src_mask):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        
        if self.norm is not None:
            x = self.norm(x)

        return x
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.proj(x)
        x = F.log_softmax(x, dim=-1)
        return x

class Transformer(nn.Module):
    def __init__(self, 
        d_model, 
        num_heads, 
        num_encoder_layers,
        num_decoder_layers,
        d_ff,
        src_vocab_size,
        tgt_vocab_size,
        dropout=0.1,
        norm_first=True,
    ):
        super(Transformer, self).__init__()

        self.src_embedding = nn.Sequential(
            Embeddings(src_vocab_size, d_model),
            PositionalEncoding(d_model, dropout),
        )
        self.tgt_embedding = nn.Sequential(
            Embeddings(tgt_vocab_size, d_model),
            PositionalEncoding(d_model, dropout),
        )

        encoder_layer = EncoderLayer(
            d_model, num_heads, d_ff, dropout, norm_first,
        )
        encoder_norm = LayerNorm(d_model)
        self.encoder = Encoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = DecoderLayer(
            d_model, num_heads, d_ff, dropout, norm_first,
        )
        decoder_norm = LayerNorm(d_model)
        self.decoder = Decoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.generator = Generator(d_model, tgt_vocab_size)

        self.config = [d_model, num_heads, num_encoder_layers, num_decoder_layers, 
                       d_ff, src_vocab_size, tgt_vocab_size, dropout, norm_first]
        
        self._init_parameters()
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        return output

    def encode(self, src, src_mask):
        src_embed = self.src_embedding(src)
        memory = self.encoder(src_embed, src_mask)
        return memory
    
    def decode(self, tgt, memory, tgt_mask, src_mask):
        tgt_embed = self.tgt_embedding(tgt)
        output = self.decoder(tgt_embed, memory, tgt_mask, src_mask)
        return output
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def save(self, path):
        params = {
            "config": self.config,
            "state_dict": self.state_dict(),
        }
        torch.save(params, path)

    @staticmethod
    def load(model_path):
        params = torch.load(model_path, map_location="cpu")
        model = Transformer(*params["config"])
        model.load_state_dict(params["state_dict"])
        return model
    

    
    


    


