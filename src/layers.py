"""Assortment of layers for use in models.py.

Author:
    Chris Chute (chute@stanford.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from util import myprint, get_available_devices

import yaml
with open('src/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config_char_embed = config['char_embed']
config_qanet = config['qanet']
model_dim = config['model_dim']

device, gpu_ids = get_available_devices()


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv layers are used in the QANet instead of
    the traditional conv layers.
    
    Based on paper
    https://arxiv.org/pdf/1610.02357.pdf
    with parameters referring to the QANet paper
    https://arxiv.org/pdf/1804.09541.pdf
    
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channel,
                                        out_channels=in_channel,
                                        kernel_size=kernel_size,
                                        groups=in_channel)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channel,
                                        out_channels=out_channel,
                                        kernel_size=1)
    
    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class PositionalEncoding(nn.Module):
    """Positional encoding layer.
    
    Args:
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
    """
    def __init__(self, length, dim):
        super(PositionalEncoding, self).__init__()
        # implementatino following https://github.com/heliumsea/QANet-pytorch/blob/master/models.py
        # freq = torch.Tensor([
        #     10000 ** (-i / dim) if i % 2 == 0 else -10000 ** ((1 - i) / dim) for i in range(dim)
        # ]).unsqueeze(1)
        # phases = torch.Tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(dim)]).unsqueeze(1)
        # pos = torch.arange(length).repeat(dim, 1).to(torch.float)
        # self.pe = nn.Parameter(torch.sin(torch.add(torch.mul(pos, freq), phases)), requires_grad=False)
        
        # implementation following https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        self.pe = torch.zeros(length, dim).to(device)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        # self.pe.transpose_(0, 1)
        # myprint('position * div_term shape', (position * div_term).shape)
        
        # myprint("self.pe shape", self.pe.shape)
        # myprint("self.pe", self.pe)
        
    def forward(self, x):
        with torch.no_grad():
            myprint('x shape', x.shape)
            pos_encoding = self.pe[:x.size(1)]
            myprint('pos enc shape:', pos_encoding.shape)
            x = x + pos_encoding
        return x


class MHA(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    Copied from assignment 4.
    """

    def __init__(self,
                 dim,
                 drop_prob=0.):
        super().__init__()
        assert dim % config_qanet['num_heads'] == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(dim, dim)
        self.query = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        # regularization
        self.attn_drop = nn.Dropout(drop_prob)
        self.resid_drop = nn.Dropout(drop_prob)
        # output projection
        self.proj = nn.Linear(dim, dim)
        # # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.n_head = config_qanet['num_heads']

    def forward(self, x, mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask[:,:,:T,:T] == 0, -1e10) # todo: just use float('-inf') instead?
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class EmbeddingWord(nn.Module):
    """Embedding layer used by BiDAF, without the character-level component.

    Word-level embeddings are further refined using a 2-layer Highway Encoder
    (see `HighwayEncoder` class for details).

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, word_vectors, hidden_size, drop_prob):
        super(EmbeddingWord, self).__init__()
        self.drop_prob = drop_prob
        self.embed_word = nn.Embedding.from_pretrained(word_vectors)
        self.proj = nn.Linear(word_vectors.size(1), hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x):
        emb = self.embed_word(x)   # (batch_size, seq_len, embed_size)
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class EmbeddingChar(EmbeddingWord):
    """Embedding layer used by BiDAF, WITH the character-level component.

    Args:
        char_vectors (torch.Tensor): Pre-trained char vectors.
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
    """
    def __init__(self, char_vectors, char_conv_kernel, word_vectors, hidden_size, drop_prob):
        super(EmbeddingChar, self).__init__(word_vectors, hidden_size, drop_prob)
        self.embed_char = nn.Embedding.from_pretrained(char_vectors)
        self.conv_char = nn.Sequential(
            nn.Conv2d(char_vectors.shape[1], hidden_size, (1, char_conv_kernel)),
            nn.ReLU()
        )
        self.hwy = HighwayEncoder(2, hidden_size * 2)   # hidden_size * 2 = [char + word]


    def forward(self, w_idxs, c_idxs):
        # get char embedding
        def embed_char_layer(x):
            emb = self.embed_char(x)    # (batch_size, seq_len, max_word_len, char_embed_size)
            emb = emb.permute(0, 3, 1, 2)
            emb = self.conv_char(emb)   # (batch_size, hidden_size, seq_len, conv_kernel_width)
            emb = F.max_pool2d(emb, (1, emb.size(-1)))
            emb = emb.permute(0, 2, 1, 3).squeeze_(-1)   # (batch_size, seq_len, hidden_size)
            return emb
        emb_char = embed_char_layer(c_idxs)
        emb_char = F.dropout(emb_char, self.drop_prob, self.training)

        # get word embedding
        emb_word = self.embed_word(w_idxs)   # (batch_size, seq_len, embed_size)
        emb_word = F.dropout(emb_word, self.drop_prob, self.training)
        emb_word = self.proj(emb_word)  # (batch_size, seq_len, hidden_size)

        emb = torch.cat([emb_char, emb_word], dim=-1)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        return emb


class HighwayEncoder(nn.Module):
    """Encode an input sequence using a highway network.

    Based on the paper:
    "Highway Networks"
    by Rupesh Kumar Srivastava, Klaus Greff, Jürgen Schmidhuber
    (https://arxiv.org/abs/1505.00387).

    Args:
        num_layers (int): Number of layers in the highway encoder.
        hidden_size (int): Size of hidden activations.
    """
    def __init__(self, num_layers, hidden_size, dim=model_dim):
        super(HighwayEncoder, self).__init__()
        self.transforms = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                         for _ in range(num_layers)])
        self.gates = nn.ModuleList([nn.Linear(hidden_size, hidden_size)
                                    for _ in range(num_layers)])
        # self.transforms = nn.ModuleList([nn.Linear(hidden_size, dim)] +
        #                                 [nn.Linear(dim, dim) for _ in range(num_layers - 1)])
        # self.gates = nn.ModuleList([nn.Linear(hidden_size, dim)] +
        #                            [nn.Linear(dim, dim) for _ in range(num_layers - 1)])

    def forward(self, x):
        for gate, transform in zip(self.gates, self.transforms):
            # Shapes of g, t, and x are all (batch_size, seq_len, hidden_size)
            g = torch.sigmoid(gate(x))
            t = F.relu(transform(x))
            x = g * t + (1 - g) * x

        return x


class RNNEncoder(nn.Module):
    """General-purpose layer for encoding a sequence using a bidirectional RNN.

    Encoded output is the RNN's hidden state at each position, which
    has shape `(batch_size, seq_len, hidden_size * 2)`.

    Args:
        input_size (int): Size of a single timestep in the input.
        hidden_size (int): Size of the RNN hidden state.
        num_layers (int): Number of layers of RNN cells to use.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 drop_prob=0.):
        super(RNNEncoder, self).__init__()
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=True,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, lengths):
        # Save original padded length for use by pad_packed_sequence
        orig_len = x.size(1)

        # Sort by length and pack sequence for RNN
        lengths, sort_idx = lengths.sort(0, descending=True)
        x = x[sort_idx]     # (batch_size, seq_len, input_size)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True)

        # Apply RNN
        self.rnn.flatten_parameters()
        x, _ = self.rnn(x)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpack and reverse sort
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=orig_len)
        _, unsort_idx = sort_idx.sort(0)
        x = x[unsort_idx]   # (batch_size, seq_len, 2 * hidden_size)

        # Apply dropout (RNN applies dropout after all but the last layer)
        x = F.dropout(x, self.drop_prob, self.training)

        return x


class BiDAFAttention(nn.Module):
    """Bidirectional attention originally used by BiDAF.

    Bidirectional attention computes attention in two directions:
    The context attends to the query and the query attends to the context.
    The output of this layer is the concatenation of [context, c2q_attention,
    context * c2q_attention, context * q2c_attention]. This concatenation allows
    the attention vector at each timestep, along with the embeddings from
    previous layers, to flow through the attention layer to the modeling layer.
    The output has shape (batch_size, context_len, 8 * hidden_size).

    Args:
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob=0.1):
        super(BiDAFAttention, self).__init__()
        self.drop_prob = drop_prob
        self.c_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.q_weight = nn.Parameter(torch.zeros(hidden_size, 1))
        self.cq_weight = nn.Parameter(torch.zeros(1, 1, hidden_size))
        for weight in (self.c_weight, self.q_weight, self.cq_weight):
            nn.init.xavier_uniform_(weight)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, c, q, c_mask, q_mask):
        batch_size, c_len, _ = c.size()
        q_len = q.size(1)
        s = self.get_similarity_matrix(c, q)        # (batch_size, c_len, q_len)
        c_mask = c_mask.view(batch_size, c_len, 1)  # (batch_size, c_len, 1)
        q_mask = q_mask.view(batch_size, 1, q_len)  # (batch_size, 1, q_len)
        s1 = masked_softmax(s, q_mask, dim=2)       # (batch_size, c_len, q_len)
        s2 = masked_softmax(s, c_mask, dim=1)       # (batch_size, c_len, q_len)

        # (bs, c_len, q_len) x (bs, q_len, hid_size) => (bs, c_len, hid_size)
        a = torch.bmm(s1, q)
        # (bs, c_len, c_len) x (bs, c_len, hid_size) => (bs, c_len, hid_size)
        b = torch.bmm(torch.bmm(s1, s2.transpose(1, 2)), c)

        x = torch.cat([c, a, c * a, c * b], dim=2)  # (bs, c_len, 4 * hid_size)

        return x

    def get_similarity_matrix(self, c, q):
        """Get the "similarity matrix" between context and query (using the
        terminology of the BiDAF paper).

        A naive implementation as described in BiDAF would concatenate the
        three vectors then project the result with a single weight matrix. This
        method is a more memory-efficient implementation of the same operation.

        See Also:
            Equation 1 in https://arxiv.org/abs/1611.01603
        """
        c_len, q_len = c.size(1), q.size(1)
        c = F.dropout(c, self.drop_prob, self.training)  # (bs, c_len, hid_size)
        q = F.dropout(q, self.drop_prob, self.training)  # (bs, q_len, hid_size)

        # Shapes: (batch_size, c_len, q_len)
        s0 = torch.matmul(c, self.c_weight).expand([-1, -1, q_len])
        s1 = torch.matmul(q, self.q_weight).transpose(1, 2)\
                                           .expand([-1, c_len, -1])
        s2 = torch.matmul(c * self.cq_weight, q.transpose(1, 2))
        s = s0 + s1 + s2 + self.bias

        return s


class BiDAFOutput(nn.Module):
    """Output layer used by BiDAF for question answering.

    Computes a linear transformation of the attention and modeling
    outputs, then takes the softmax of the result to get the start pointer.
    A bidirectional LSTM is then applied the modeling output to produce `mod_2`.
    A second linear+softmax of the attention output and `mod_2` is used
    to get the end pointer.

    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(BiDAFOutput, self).__init__()
        self.att_linear_1 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_1 = nn.Linear(2 * hidden_size, 1)

        self.rnn = RNNEncoder(input_size=2 * hidden_size,
                              hidden_size=hidden_size,
                              num_layers=1,
                              drop_prob=drop_prob)

        self.att_linear_2 = nn.Linear(8 * hidden_size, 1)
        self.mod_linear_2 = nn.Linear(2 * hidden_size, 1)

    def forward(self, att, mod, mask):
        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.att_linear_1(att) + self.mod_linear_1(mod)
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
