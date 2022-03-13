"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import sample_layers
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import myprint
from args import get_train_args, get_test_args
import yaml

# args = get_train_args()
args = get_test_args()
use_char_embed = args.use_char_embed
use_qanet = args.use_qanet
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


class sampleQANet(nn.Module):
    '''Sample codebase copied from
    https://github.com/BangLiu/QANet-PyTorch/blob/master/model/QANet.py
    '''
    def __init__(self, char_vectors, word_vectors, drop_prob=0.):
        super(sampleQANet, self).__init__()
        if use_char_embed:
            self.emb = layers.EmbeddingChar(char_vectors=char_vectors,
                                            char_conv_kernel=config_char_embed['char_conv_kernel'],
                                            word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
            
            self.qanet_c_emb = sample_layers.EncoderBlock(conv_num=4,
                                                          d_model=200,
                                                          num_head=10,
                                                          k=7,
                                                          dropout=drop_prob)
                        
        else:
            self.emb = layers.EmbeddingWord(word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
                        
            self.enc = layers.RNNEncoder(input_size=hidden_size,
                                        hidden_size=hidden_size,
                                        num_layers=1,
                                        drop_prob=drop_prob)
        
        # self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
        #                                  drop_prob=drop_prob)
        self.cq_att = sample_layers.CQAttention(config['model_dim'])
        
        # resize layer: self.att [batch, seq_len, model_dim * 4] --> [batch, seq_len, model_dim]
        self.att_resize = sample_layers.Initialized_Conv1d(config['model_dim'] * 4, config['model_dim'])
        
        # QANet model encoder layers
        enc_block = sample_layers.EncoderBlock(conv_num=2,
                                                d_model=config['model_dim'],
                                                num_head=10,
                                                k=5,
                                                dropout=drop_prob)
        self.mod_blocks = nn.ModuleList([enc_block] * config_qanet['model_block_num'])
        
        # QANet output layer
        self.out = sample_layers.Pointer(config['model_dim'])
        

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

        c_enc = self.qanet_c_emb(c_emb.transpose(1, 2), c_mask, 1, 1) #.transpose(1, 2)
        # q_enc = self.qanet_q_emb(q_emb, q_mask)
        q_enc = self.qanet_c_emb(q_emb.transpose(1, 2), q_mask, 1, 1) #.transpose(1, 2)
        
        if debugging:
            myprint("c_enc", c_enc.size())
        
        att = self.cq_att(c_enc, q_enc, c_mask, q_mask)
        
        out = self.att_resize(att)
        for i, mod_block in enumerate(self.mod_blocks):
            out = mod_block(out, c_mask, i * (2+2)+1, 7)
        out_1 = out
        for i, mod_block in enumerate(self.mod_blocks):
            out = mod_block(out, c_mask, i * (2+2)+1, 7)
        out_2 = out
        for i, mod_block in enumerate(self.mod_blocks):
            out = mod_block(out, c_mask, i * (2+2)+1, 7)
        out_3 = out
        
        out = self.out(out_1, out_2, out_3, c_mask)
        
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
            
            self.enc = QANetEncoderBlock(length=config['para_limit'],
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
        # self.att = sample_layers.CQAttention(config['model_dim'], drop_prob)
        
        # resize layer: self.att [batch, seq_len, model_dim * 4] --> [batch, seq_len, model_dim]
        self.att_resize = layers.DepthwiseSeparableConv(config['model_dim'] * 4, config['model_dim'])
        
        # QANet model encoder layers
        # self.mod_block = QANetEncoderBlock(length=config['para_limit'],
        #                                    conv_layer_num=config_qanet['model_conv_layer_num'],
        #                                    model_dim=config['model_dim'],
        #                                    drop_prob=drop_prob)
        # self.mod_blocks = nn.ModuleList([self.mod_block] * config_qanet['model_block_num'])
        mod_block1 = QANetEncoderBlock(length=config['para_limit'],
                                       conv_layer_num=config_qanet['model_conv_layer_num'],
                                       model_dim=config['model_dim'],
                                       drop_prob=drop_prob)
        mod_block2 = QANetEncoderBlock(length=config['para_limit'],
                                       conv_layer_num=config_qanet['model_conv_layer_num'],
                                       model_dim=config['model_dim'],
                                       drop_prob=drop_prob)
        mod_block3 = QANetEncoderBlock(length=config['para_limit'],
                                       conv_layer_num=config_qanet['model_conv_layer_num'],
                                       model_dim=config['model_dim'],
                                       drop_prob=drop_prob)
        mod_block4 = QANetEncoderBlock(length=config['para_limit'],
                                       conv_layer_num=config_qanet['model_conv_layer_num'],
                                       model_dim=config['model_dim'],
                                       drop_prob=drop_prob)
        mod_block5 = QANetEncoderBlock(length=config['para_limit'],
                                       conv_layer_num=config_qanet['model_conv_layer_num'],
                                       model_dim=config['model_dim'],
                                       drop_prob=drop_prob)
        mod_block6 = QANetEncoderBlock(length=config['para_limit'],
                                       conv_layer_num=config_qanet['model_conv_layer_num'],
                                       model_dim=config['model_dim'],
                                       drop_prob=drop_prob)
        mod_block7 = QANetEncoderBlock(length=config['para_limit'],
                                       conv_layer_num=config_qanet['model_conv_layer_num'],
                                       model_dim=config['model_dim'],
                                       drop_prob=drop_prob)

        self.mod_blocks = nn.ModuleList([
            mod_block1, mod_block2, mod_block3, mod_block4, mod_block5, mod_block6, mod_block7
            ])

        # QANet output layer
        self.out = layers.Pointer(drop_prob=drop_prob)
        # self.out = sample_layers.Pointer(config['model_dim'])
        
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

        c_enc = self.enc(c_emb, c_mask, 1, 1)
        q_enc = self.enc(q_emb, q_mask, 1, 1)
        
        if debugging:
            myprint("c_enc", c_enc.size())
        
        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        # att = self.att(c_enc.transpose(1, 2), q_enc.transpose(1, 2), c_mask, q_mask).transpose(1, 2)
        
        out = self.att_resize(att.transpose(1, 2)).transpose(1, 2)
        # for i, mod_block in enumerate(self.mod_blocks):
        #     out1 = mod_block(out, c_mask,
        #                      i * (mod_block.conv_layer_num + 1) + 1, len(self.mod_blocks))
        # out2 = out1
        # for i, mod_block in enumerate(self.mod_blocks):
        #     out2 = mod_block(out2, c_mask,
        #                      i * (mod_block.conv_layer_num + 1) + 1, len(self.mod_blocks))
        # out3 = out2
        # for i, mod_block in enumerate(self.mod_blocks):
        #     out3 = mod_block(out3, c_mask,
        #                      i * (mod_block.conv_layer_num + 1) + 1, len(self.mod_blocks))
        for i, mod_block in enumerate(self.mod_blocks):
            out = mod_block(out, c_mask,
                            i * (mod_block.conv_layer_num + 1) + 1, len(self.mod_blocks))
        out1 = out
        for i, mod_block in enumerate(self.mod_blocks):
            out = mod_block(out, c_mask,
                            i * (mod_block.conv_layer_num + 1) + 1, len(self.mod_blocks))
        out2 = out
        for i, mod_block in enumerate(self.mod_blocks):
            out = mod_block(out, c_mask,
                            i * (mod_block.conv_layer_num + 1) + 1, len(self.mod_blocks))
        out3 = out
        
        # out = self.out(out1, out2, out3, c_mask)
        out = self.out(out1.transpose(1, 2), out2.transpose(1, 2), out3.transpose(1, 2), c_mask)

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
        self.conv_layer_num = conv_layer_num
        self.convs = nn.ModuleList([layers.DepthwiseSeparableConv(model_dim, model_dim)
                                    for _ in range(conv_layer_num)])
        # self.convs = nn.ModuleList([sample_layers.DepthwiseSeparableConv(model_dim, model_dim, 5)
        #                             for _ in range(self.conv_layer_num)])
        self.mha = layers.MHA(dim=model_dim, drop_prob=drop_prob)
        # self.mha = sample_layers.SelfAttention(model_dim, config_qanet['num_heads'], drop_prob)
        self.layer_norm_convs = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(self.conv_layer_num)])
        self.layer_norm_att = nn.LayerNorm(model_dim)
        self.layer_norm_forward = nn.LayerNorm(model_dim)
        # self.mlp = nn.Sequential(nn.Linear(model_dim, model_dim),
        #                          nn.ReLU(),
        #                          nn.Linear(model_dim, model_dim))
        self.fc1 = layers.Initialized_Conv1d(model_dim, model_dim, relu=True, bias=True)
        self.fc2 = layers.Initialized_Conv1d(model_dim, model_dim, bias=True)
        
    def forward(self, x, mask, l, num_blocks):
        total_layers = (len(self.layer_norm_convs) + 2) * num_blocks  # total # of layers in one encoder block
        out = self.pos_encoder(x) # [batch, seq_len, model_dim]
        # x = sample_layers.PosEncoder(x.transpose(1, 2)).transpose(1, 2)
        # sub block 1: layernorm + conv
        for i, conv in enumerate(self.convs):
            res = out
            # if debugging: myprint('before layernorm - x shape', x.size())
            
            out = self.layer_norm_convs[i](out)
            # if debugging: myprint('after layernorm - x shape', x.size())
            if i % 2 == 0:
                out = F.dropout(out, self.drop_prob, self.training) # * float(l) / total_layers, self.training)
            
            out = conv(out.transpose(1, 2)).transpose(1, 2)
            if debugging: myprint('after conv - x shape', x.size())
            out = self.layer_dropout(out, res, self.drop_prob * float(l) / total_layers)
            l += 1

        # sub block 2: layernorm + self attention
        res = out
        out = self.layer_norm_att(out)
        # x = F.dropout(x, self.drop_prob, self.training)
        out = self.mha(out, mask)# + res
        # x = self.mha(x.transpose(1, 2), mask).transpose(1, 2)# + res
        l += 1
        if debugging: myprint('after att - x shape', out.size())
        out = self.layer_dropout(out, res, self.drop_prob * float(l) / total_layers)
        
        # sub block 3: layernorm + feed forward
        res = out
        out = self.layer_norm_forward(out)
        # x = F.dropout(x, self.drop_prob, self.training)
        # x = self.mlp(x)# + res
        out = out.transpose(1, 2)
        out = self.fc1(out)
        out = self.fc2(out)
        out = out.transpose(1, 2)# + res
        if debugging: myprint('after mlp - x shape', out.size())
        out = self.layer_dropout(out, res, self.drop_prob * float(l) / total_layers)
        
        return out
    
    def layer_dropout(self, inputs, residual, dropout):
        return F.dropout(inputs, dropout, self.training) + residual
        # if self.training:
        #     pred = torch.empty(1).uniform_(0,1) < dropout
        #     if pred:
        #         return residual
        #     else:
        #     return F.dropout(inputs, dropout, training=self.training) + residual
        # else:
        #     return inputs + residual