"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F
from args import get_train_args
import yaml

args = get_train_args()
use_char_embed = args.use_char_embed
use_qanet = args.use_qanet
hidden_size = args.hidden_size

with open('src/config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config_char_embed = config['char_embed']
config_qanet = config['qanet']

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, char_vectors, word_vectors, drop_prob=0.):
        super(BiDAF, self).__init__()
        if use_char_embed:
            self.emb = layers.EmbeddingChar(char_vectors=char_vectors,
                                            char_conv_kernel=config_char_embed['char_conv_kernel'],
                                            word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
            
            if use_qanet:
                self.qanet_emb_enc = QANetEncoderBlock(
                    length=config_qanet['emb_length'],
                    conv_layer_num=config_qanet['emb_conv_layer_num'],
                    channel=hidden_size*2,
                    conv_kernel_size=config_qanet['emb_pointwise_conv_kernel'],
                    embed_dim=config_qanet['emb_conv_kernel_size'],
                    num_heads=config_qanet['emb_num_heads'],
                    drop_prob=drop_prob)
                
                self.qanet_enc_block = QANetEncoderBlock(
                    length=config_qanet['model_length'],
                    conv_layer_num=config_qanet['model_conv_layer_num'],
                    channel=hidden_size*2,
                    conv_kernel_size=config_qanet['model_pointwise_conv_kernel'],
                    embed_dim=config_qanet['model_conv_kernel_size'],
                    num_heads=config_qanet['model_num_heads'],
                    drop_prob=drop_prob)
                
                self.qanet_model_enc = nn.ModuleList([self.qanet_enc_block] * 7)
            
            else:
                self.enc = layers.RNNEncoder(input_size=hidden_size*2,
                                            hidden_size=hidden_size,
                                            num_layers=1,
                                            drop_prob=drop_prob)

        else:
            self.emb = layers.EmbeddingWord(word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
            
            if use_qanet:
                self.qanet_emb_enc = QANetEncoderBlock(
                    length=config_qanet['emb_length'],
                    conv_layer_num=config_qanet['emb_conv_layer_num'],
                    channel=hidden_size,
                    conv_kernel_size=config_qanet['pointwise_conv_kernel'],
                    model_dim=config_qanet['model_dim'],
                    num_heads=config_qanet['num_heads'],
                    drop_prob=drop_prob)
                
                self.qanet_enc_block = QANetEncoderBlock(
                    length=config_qanet['model_length'],
                    conv_layer_num=config_qanet['model_conv_layer_num'],
                    channel=hidden_size,
                    conv_kernel_size=config_qanet['pointwise_conv_kernel'],
                    model_dim=config_qanet['model_dim'],
                    num_heads=config_qanet['num_heads'],
                    drop_prob=drop_prob)
                
                self.qanet_model_enc = nn.ModuleList([self.qanet_enc_block] * 7)
            
            else:
                self.enc = layers.RNNEncoder(input_size=hidden_size,
                                            hidden_size=hidden_size,
                                            num_layers=1,
                                            drop_prob=drop_prob)
        

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        # TODO: Add char embedding.
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        if use_char_embed:
            c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size * 2)
            q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size * 2)
        else:
            c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class QANetEncoderBlock(nn.Module):
    # TODO: check dimensions
    """Encoder block used in the QANet.
    
    Based on the QANet paper
    https://arxiv.org/pdf/1804.09541.pdf
    
    """
    def __init__(self,
                 length,
                 conv_layer_num,
                 channel,
                 conv_kernel_size,
                 model_dim,
                 num_heads,
                 drop_prob=0.):
        super(QANetEncoderBlock, self).__init__()
        self.drop_prob = drop_prob
        self.pos_encoder = layers.PositionalEncoding(length, model_dim)
        self.convs = nn.ModuleList([
            layers.DepthwiseSeparableConv(channel, channel, conv_kernel_size) 
            for _ in range(conv_layer_num)])
        self.mha = nn.MultiheadAttention(model_dim, num_heads, dropout=self.drop_prob)
        self.layer_norm = nn.LayerNorm([model_dim, length])
        self.mlp = nn.Linear(channel, channel)
        
    def forward(self, x, mask):
        x = self.pos_encoder(x)
        # sub block 1: layernorm + conv
        for i, conv in enumerate(self.convs):
            res = x.copy()
            x = self.layer_norm(x)
            x = conv(x)
            x = F.relu(x) + res
            x = F.dropout(x, self.drop_prob, self.training)
        # sub block 2: layernorm + self attention
        res = x.copy()
        x = self.layer_norm(x)
        x = F.relu(self.mha(x)) + res
        x = F.dropout(x, self.drop_prob, self.training)
        # sub block 3: layernorm + feed forward
        res = x.copy()
        x = self.layer_norm(x)
        x = F.relu(self.mlp(x)) + res
        x = F.dropout(x, self.drop_prob, self.training)
