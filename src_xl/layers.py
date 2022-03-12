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
with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config_char_embed = config['char_embed']
config_qanet = config['qanet']
model_dim = config['model_dim']

from args import get_train_args
args = get_train_args()
debugging = False

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
                 kernel_size=config_qanet['pointwise_conv_kernel']):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_channel,
                                        out_channels=in_channel,
                                        kernel_size=kernel_size,
                                        padding=kernel_size // 2,
                                        groups=in_channel,
                                        bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_channel,
                                        out_channels=out_channel,
                                        kernel_size=1)
    
    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


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
        # self.pe.transpose_(0, 1)
        
        # implementation following https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        position = torch.arange(length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, 1, dim).to(device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # with torch.no_grad():
        if debugging: myprint('x shape', x.shape)
        out = x.transpose(0, 1)
        if debugging: myprint('x shape', out.shape)
        
        pos_encoding = self.pe[:out.size(0)]
        if debugging:
            myprint('pos enc shape', pos_encoding.shape)
            myprint('pos enc', pos_encoding)
        out = (out + pos_encoding).transpose(0, 1)
        if debugging:
            myprint('pos enc out shape', out.size())
            myprint('pos enc out[0]', out[0])
        
        return out


class MHA(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    Copied from assignment 4.
    """

    def __init__(self,
                 dim,
                 drop_prob=0.):
        super(MHA, self).__init__()
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
        if debugging: myprint("mha x size", x.size())
        
        if debugging:
            myprint("mha input x size", x.size())
            myprint("mha input mask size", mask.size())
            myprint("mha input x", x)
            myprint("mha input mask", mask)
        mask = mask.view(B, 1, 1, T).to(torch.float)
        if debugging:
            myprint("mha reshaped mask size", mask.size())
            myprint("mha reshaped mask", mask)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # if debugging: myprint("mha k size", k.size())

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if debugging:
            myprint("mha att size before mask", att.size())
            myprint("mha att before mask", att)
        att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        if debugging:
            myprint("mha att size after mask", att.size())
            myprint("mha att after mask", att)
        
        att = F.softmax(att, dim=-1)
        if debugging:
            myprint("mha att after softmax", att)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

    
class Pointer(nn.Module):
    def __init__(self, drop_prob=0.):
        super(Pointer, self).__init__()
        self.w_start = nn.Linear(model_dim * 2, 1, bias=False)
        self.w_end = nn.Linear(model_dim * 2, 1, bias=False)
        # initialization
        # lim = math.sqrt(3 / (2 * model_dim))
        # nn.init.uniform_(self.w_start.weight.data, -lim, lim)
        # nn.init.uniform_(self.w_end.weight.data, -lim, lim)
        
    def forward(self, x1, x2, x3, mask):
        x_start = torch.cat([x1, x2], dim=-1)
        x_end = torch.cat([x2, x3], dim=-1)
        if debugging:
            myprint('x1', x1)
            myprint('x2', x2)
            myprint('x3', x3)
            myprint('x_start shape', x_start.size())
        logits_1 = self.w_start(x_start)
        logits_2 = self.w_end(x_end)
        if debugging:
            myprint('logits_1 size', logits_1.size())
            myprint('logits_1', logits_1)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(-1), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(-1), mask, log_softmax=True)
        if debugging:
            myprint('log_p1 size', log_p1.size())
            myprint('log_p1', log_p1)

        return log_p1, log_p2


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
        if debugging:
            myprint('input mask', mask)
            myprint('logits_1 size', logits_1.size())
            myprint('logits_1', logits_1)
            
        mod_2 = self.rnn(mod, mask.sum(-1))
        logits_2 = self.att_linear_2(att) + self.mod_linear_2(mod_2)

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(), mask, log_softmax=True)

        return log_p1, log_p2
    

class QANetEncoderBlock(nn.Module):
    # TODO: check dimensions
    """Encoder block used in the QANet.
    
    Based on the QANet paper
    https://arxiv.org/pdf/1804.09541.pdf
    
    """
    def __init__(self,
                 length,
                 conv_layer_num,
                 model_dim,
                 drop_prob=0.):
        super(QANetEncoderBlock, self).__init__()
        self.drop_prob = drop_prob
        self.pos_encoder = PositionalEncoding(length, model_dim)
        self.conv_layer_num = conv_layer_num
        self.convs = nn.ModuleList([DepthwiseSeparableConv(model_dim, model_dim)
                                    for _ in range(conv_layer_num)])
        # self.convs = nn.ModuleList([sample_layers.DepthwiseSeparableConv(model_dim, model_dim, 5)
        #                             for _ in range(self.conv_layer_num)])
        self.mha = MHA(dim=model_dim, drop_prob=drop_prob)
        # self.mha = sample_layers.SelfAttention(model_dim, config_qanet['num_heads'], drop_prob)
        self.layer_norm_convs = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(self.conv_layer_num)])
        self.layer_norm_att = nn.LayerNorm(model_dim)
        self.layer_norm_forward = nn.LayerNorm(model_dim)
        self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim),
                                 nn.ReLU(),
                                 nn.Linear(model_dim, model_dim))
        # self.fc1 = sample_layers.Initialized_Conv1d(model_dim, model_dim, relu=True, bias=True)
        # self.fc2 = sample_layers.Initialized_Conv1d(model_dim, model_dim, bias=True)
        
    def forward(self, x, mask, l, num_blocks):
        total_layers = (len(self.layer_norm_convs) + 2) * num_blocks  # total # of layers in one encoder block
        x = self.pos_encoder(x) # [batch, seq_len, model_dim]
        # x = sample_layers.PosEncoder(x.transpose(1, 2)).transpose(1, 2)
        # sub block 1: layernorm + conv
        for i, conv in enumerate(self.convs):
            res = x
            # if debugging: myprint('before layernorm - x shape', x.size())
            
            x = self.layer_norm_convs[i](x)
            # if debugging: myprint('after layernorm - x shape', x.size())
            if i % 2 == 0:
                x = F.dropout(x, self.drop_prob, self.training) # * float(l) / total_layers, self.training)
            
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            if debugging: myprint('after conv - x shape', x.size())
            x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
            l += 1

        # sub block 2: layernorm + self attention
        res = x
        x = self.layer_norm_att(x)
        x = F.dropout(x, self.drop_prob, self.training)
        x = self.mha(x, mask)# + res
        # x = self.mha(x.transpose(1, 2), mask).transpose(1, 2)# + res
        l += 1
        if debugging: myprint('after att - x shape', x.size())
        x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
        
        # sub block 3: layernorm + feed forward
        res = x
        x = self.layer_norm_forward(x)
        x = F.dropout(x, self.drop_prob, self.training)
        x = self.mlp(x)# + res
        # x = x.transpose(1, 2)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = x.transpose(1, 2)# + res
        if debugging: myprint('after mlp - x shape', x.size())
        x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
        
        return x
    
    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual
    
    
# transformer XL
class QAXLEncoderBlock(nn.Module):
    def __init__(self, conv_layer_num, n_head, model_dim, head_dim, dropout = 0.1):
        super(QAXLEncoderBlock, self).__init__()       
        self.drop_prob = dropout
        self.conv_layer_num = conv_layer_num
        self.convs = nn.ModuleList([DepthwiseSeparableConv(model_dim, model_dim)
                                    for _ in range(conv_layer_num)])
        self.layer_norm_convs = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(self.conv_layer_num)])
        
        self.att = RelPartialLearnableMultiHeadAttn(n_head, model_dim, head_dim, self.drop_prob, pre_lnorm = True)
        self.layer_norm_att = nn.LayerNorm(model_dim)
        self.layer_norm_forward = nn.LayerNorm(model_dim)
#         self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim*4),
#                                  nn.ReLU(),
#                                  nn.Dropout(self.drop_prob),
#                                  nn.Linear(model_dim*4, model_dim))
        self.fc1 = Initialized_Conv1d(model_dim, model_dim, relu=True, bias=True)
        self.fc2 = Initialized_Conv1d(model_dim, model_dim, bias=True)


    def forward(self, x, mask, l, num_blocks, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        """
        dropout probability: uses stochastic depth survival probability = 1 - (l/L)*pL, 
        reference here: https://arxiv.org/pdf/1603.09382.pdf 
        """   
        total_layers = (len(self.layer_norm_convs) + 2) * num_blocks  # total # of layers in one encoder block

        for i, conv in enumerate(self.convs):
            res = x        
            x = self.layer_norm_convs[i](x)
            if i % 2 == 0:
                x = F.dropout(x, self.drop_prob, self.training) # * float(l) / total_layers, self.training)
            
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
            l += 1

        res = x
        x = self.layer_norm_att(x)
        x = F.dropout(x, self.drop_prob, self.training)
        x = self.att(x, mask, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems) # modified
        l += 1
        x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
        
        res = x
        x = self.layer_norm_forward(x)
        x = F.dropout(x, self.drop_prob, self.training)
        x = x.transpose(1, 2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.transpose(1, 2)# + res

        x = self.layer_dropout(x, res, self.drop_prob * float(l) / total_layers)
        return x

        
    def layer_dropout(self, inputs, residual, dropout):
        if self.training:
            pred = torch.empty(1).uniform_(0,1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual

class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)
        
class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq) # outer product
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]

def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)
        
        
class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, pre_lnorm = True):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)
        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def forward(self, w, mask, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
#         w: batch_size, q_len, hidden_size*2
#         mems: q_len, batch_size, hidden_size*2
        w = w.permute(1, 0, 2) #qlen x bsz x hidden_size, 282, 64, 200
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)
        if mems is not None:
            cat = torch.cat([mems, w], 0)
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                         # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        rr_head_q = w_head_q + r_r_bias
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
        BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        sizes = mask.size()
        mask = mask.unsqueeze(-1).unsqueeze(-1)
        mask = mask.permute(0, 2, 3, 1)
        attn_score = attn_score.permute(2, 3, 1, 0)
        attn_score = mask_logits(attn_score, mask)
        attn_score = attn_score.permute(3, 2, 0, 1)
                
        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float('inf')).type_as(attn_score)

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)
        # qlen x batch_size x (n_head*d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)  # qlen x batch_size x d_model
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            #output = w + attn_out
            output = attn_out # qlen x batch_size x d_model
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out) #(n, d_model)
            output = self.layer_norm(attn_out)

        return output.permute(1, 0, 2)