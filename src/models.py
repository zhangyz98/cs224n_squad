"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import myprint
from args import get_train_args
import yaml

args = get_train_args()
use_char_embed = args.use_char_embed
use_qanet = args.use_qanet
use_qanet_model = args.use_qanet_model
hidden_size = args.hidden_size
debugging = args.test is False

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
            
            self.enc = layers.RNNEncoder(input_size=hidden_size*2,
                                            hidden_size=hidden_size,
                                            num_layers=1,
                                            drop_prob=drop_prob)

        else:
            self.emb = layers.EmbeddingWord(word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
                        
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
        
        if debugging:
            myprint("c_enc", c_enc.size())
        
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)
        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out


class QANet(nn.Module):
    # TODO: debug QANet model
    def __init__(self, char_vectors, word_vectors, drop_prob=0.):
        super(QANet, self).__init__()
        if use_char_embed:
            self.emb = layers.EmbeddingChar(char_vectors=char_vectors,
                                            char_conv_kernel=config_char_embed['char_conv_kernel'],
                                            word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
            
            self.qanet_c_emb = QANetEncoderBlock(
                length=config['para_limit'],
                conv_layer_num=config_qanet['emb_conv_layer_num'],
                model_dim=config['model_dim'],
                drop_prob=drop_prob)
            
            self.qanet_q_emb = QANetEncoderBlock(
                length=config['ques_limit'],
                conv_layer_num=config_qanet['emb_conv_layer_num'],
                model_dim=config['model_dim'],
                drop_prob=drop_prob)
            
        else:
            self.emb = layers.EmbeddingWord(word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
                        
            self.enc = layers.RNNEncoder(input_size=hidden_size,
                                        hidden_size=hidden_size,
                                        num_layers=1,
                                        drop_prob=drop_prob)
        
        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)
        
        if use_qanet_model:
            # resize layer: self.att [batch, seq_len, model_dim * 4] --> [batch, seq_len, model_dim]
            self.att_resize = layers.DepthwiseSeparableConv(config['model_dim'] * 4, config['model_dim'])
            
            # QANet model encoder layers
            enc_block = QANetEncoderBlock(length=config['para_limit'],
                                          conv_layer_num=config_qanet['model_conv_layer_num'],
                                          model_dim=config['model_dim'],
                                          drop_prob=drop_prob)
            self.mod_blocks = nn.ModuleList([enc_block] * config_qanet['model_block_num'])
            
            # QANet output layer
            self.out = layers.Pointer(drop_prob=drop_prob)
        
        else:
            self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                         hidden_size=hidden_size,
                                         num_layers=2,
                                         drop_prob=drop_prob)

            self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                          drop_prob=drop_prob)
            

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        # TODO: Debug QANet encoders.
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        if use_char_embed:
            c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size * 2)
            q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size * 2)
        else:
            c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.qanet_c_emb(c_emb, c_mask)
        q_enc = self.qanet_q_emb(q_emb, q_mask)
        
        if debugging:
            myprint("c_enc", c_enc.size())
        
        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        
        if use_qanet_model:
            out_1 = self.att_resize(att.transpose(1, 2)).transpose(1, 2)
            for _, mod_block in enumerate(self.mod_blocks):
                out_1 = mod_block(out_1, c_mask)
            out_2 = out_1
            for _, mod_block in enumerate(self.mod_blocks):
                out_2 = mod_block(out_2, c_mask)
            out_3 = out_2
            for _, mod_block in enumerate(self.mod_blocks):
                out_3 = mod_block(out_3, c_mask)
            
            out = self.out(out_1, out_2, out_3, c_mask)
        
        else:
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
                 model_dim,
                 drop_prob=0.):
        super(QANetEncoderBlock, self).__init__()
        self.drop_prob = drop_prob
        self.pos_encoder = layers.PositionalEncoding(length, model_dim)
        self.convs = nn.ModuleList([layers.DepthwiseSeparableConv(model_dim, model_dim)
                                    for _ in range(conv_layer_num)])
        self.mha = layers.MHA(dim=model_dim, drop_prob=drop_prob)
        # self.mha = nn.MultiheadAttention(embed_dim=model_dim,
        #                                  num_heads=config_qanet['num_heads'],
        #                                  dropout=drop_prob)
        self.layer_norm_convs = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(conv_layer_num)])
        self.layer_norm_att = nn.LayerNorm(model_dim)
        self.layer_norm_forward = nn.LayerNorm(model_dim)
        self.mlp = nn.Linear(model_dim, model_dim)
        
    def forward(self, x, mask):
        x = self.pos_encoder(x) # [batch, seq_len, model_dim]
        # sub block 1: layernorm + conv
        for i, conv in enumerate(self.convs):
            res = x
            if debugging: myprint('before layernorm - x shape', x.size())
            x = self.layer_norm_convs[i](x)
            if debugging: myprint('after layernorm - x shape', x.size())
            x = conv(x.transpose(1, 2)).transpose(1, 2)
            if debugging: myprint('after conv - x shape', x.size())
            x = x + res
            x = F.dropout(x, self.drop_prob, self.training)
        
        # sub block 2: layernorm + self attention
        res = x
        x = self.layer_norm_att(x)
        x = self.mha(x, mask) + res
        if debugging: myprint('after att - x shape', x.size())
        x = F.dropout(x, self.drop_prob, self.training)
        
        # sub block 3: layernorm + feed forward
        res = x
        x = self.layer_norm_forward(x)
        x = F.relu(self.mlp(x)) + res
        if debugging: myprint('after mlp - x shape', x.size())
        x = F.dropout(x, self.drop_prob, self.training)
        
        return x
