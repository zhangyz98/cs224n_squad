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
from args import get_train_args
import yaml

args = get_train_args()
use_char_embed = args.use_char_embed
use_qanet = args.use_qanet
hidden_size = args.hidden_size
debugging = False

with open('./config.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config_char_embed = config['char_embed']
config_qanet = config['qanet']
config_qaxl = config["qaxl"]

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
    def __init__(self, char_vectors, word_vectors, drop_prob=0.1):
        super(QANet, self).__init__()
    # from QANet
        if use_char_embed:
            self.emb = layers.EmbeddingChar(char_vectors=char_vectors,
                                            char_conv_kernel=config_char_embed['char_conv_kernel'],
                                            word_vectors=word_vectors,
                                            hidden_size=hidden_size, # args.hidden_size = 100
                                            drop_prob=drop_prob)
            
        else:
            self.emb = layers.EmbeddingWord(word_vectors=word_vectors,
                                            hidden_size=hidden_size,
                                            drop_prob=drop_prob)
        
        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)      
        # resize layer: self.att [batch, seq_len, model_dim * 4] --> [batch, seq_len, model_dim]
        self.att_resize = layers.DepthwiseSeparableConv(config['model_dim'] * 4, config['model_dim'])
        
        # QANet output layer
        self.out = layers.Pointer(drop_prob=drop_prob)
        
    # for transformer XL:     
        self.enc = layers.QAXLEncoderBlock(conv_layer_num = config_qanet["emb_conv_layer_num"], 
                                           n_head = config_qaxl["num_heads"], 
                                           model_dim = config_qaxl["model_dim"], 
                                           head_dim = config_qaxl["head_dim"], 
                                           dropout=0.1)
        self.drop = nn.Dropout(0.1)
        self.pos_emb = layers.PositionalEmbedding(2*hidden_size) #hidden size?
        self.mod_blocks = nn.ModuleList([layers.QAXLEncoderBlock(conv_layer_num = config_qanet["model_conv_layer_num"], 
                                                                 n_head = config_qaxl["num_heads"], 
                                                                 model_dim = config_qaxl["model_dim"], 
                                                                 head_dim = config_qaxl["head_dim"], 
                                                                 dropout=0.1) \
                                         for _ in range(config_qanet["model_block_num"])]) # model_block_num: 5
        self.r_w_bias = nn.Parameter(torch.Tensor(config_qaxl["num_heads"], config_qaxl["head_dim"]))
        self.r_r_bias = nn.Parameter(torch.Tensor(config_qaxl["num_heads"], config_qaxl["head_dim"]))  
        self.mem_len = 64 # 256
        
        
    def enc_forward(self, emb, mask, mems=None):
        # emb: (batch_size, c_len, hidden_size * 2)
        q_len = emb.shape[1]
        m_len = mems[0].size(0) if mems is not None else 0
        k_len = m_len + q_len
        
        if self.training:
            dec_attn_mask = torch.triu(
                emb.new_ones(q_len, k_len), diagonal=1+m_len).bool()[:,:,None]
            # upper triangular mask (_, _, 1)
        else:
            all_ones = emb.new_ones(q_len, k_len)
            mask_len = k_len - self.mem_len
            if mask_len > 0:
                mask_shift_len = q_len - mask_len
            else:
                mask_shift_len = q_len
            dec_attn_mask = (torch.triu(all_ones, m_len+1)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None]
        hids = []
        pos_seq = torch.arange(k_len-1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        
#         if self.clamp_len > 0:
#             pos_seq.clamp_(max=self.clamp_len)
#         clamp_len = -1
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(emb)
        pos_emb = self.drop(pos_emb)
        
        hids.append(core_out)
        mems_i = None if mems is None else mems[1]
#         print("mems in enc_forward", mems_i.size())
        core_out = self.enc(core_out, mask, 1, 1, pos_emb, self.r_w_bias, self.r_r_bias, 
                                dec_attn_mask=dec_attn_mask, mems=mems_i)
        hids.append(core_out)
        core_out = self.drop(core_out)
#         print("len(hids) in enc_forward", len(hids))
#         print("len(mems) in enc_forward", len(mems))
        new_mems = self._update_mems(hids, mems, m_len, q_len)

        return core_out, new_mems
        
        
    def mod_forward(self, emb, mask, mems=None):
        q_len = emb.shape[1]
        m_len = mems[0].size(0) if mems is not None else 0
        k_len = m_len + q_len
        
        if self.training:
            dec_attn_mask = torch.triu(
                emb.new_ones(q_len, k_len), diagonal=1+m_len).bool()[:,:,None]
            # upper triangular mask (_, _, 1)
        else:
            all_ones = emb.new_ones(q_len, k_len)
            mask_len = k_len - self.mem_len
            if mask_len > 0:
                mask_shift_len = q_len - mask_len
            else:
                mask_shift_len = q_len
            dec_attn_mask = (torch.triu(all_ones, m_len+1)
                    + torch.tril(all_ones, -mask_shift_len)).bool()[:, :, None]
        hids = []
        pos_seq = torch.arange(k_len-1, -1, -1.0, device=emb.device, dtype=emb.dtype)
        
#         if self.clamp_len > 0:
#             pos_seq.clamp_(max=self.clamp_len)
#         clamp_len = -1
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(emb)
        pos_emb = self.drop(pos_emb)
        
        hids.append(core_out)
        for i, layer in enumerate(self.mod_blocks):
            mems_i = None if mems is None else mems[i]
#             print("mems in mod_forward", mems_i.size())
            core_out = self.enc(core_out, mask, i*(2+2)+1, 7, pos_emb, self.r_w_bias, self.r_r_bias, 
                             dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)
        core_out = self.drop(core_out)
#         print("len(hids) in mod_forward", len(hids))
#         print("len(mems) in mod_forward", len(mems))
        new_mems = self._update_mems(hids, mems, m_len, q_len) #  q_len, batch_size, hidden_size*2
        return core_out, new_mems 

        
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs, *mems):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        c_mems, q_mems, a_mems = mems # modified
        if not c_mems:
            c_mems = self._init_mems(2)
        if not q_mems:
            q_mems = self._init_mems(2)
        if not a_mems:
            a_mems = self._init_mems(6)
            
        if use_char_embed:
            c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size * 2)
            q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size * 2)
        else:
            c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)
#         print("c_mems size", c_mems[0].size())
        c_enc, e_mems = self.enc_forward(c_emb, c_mask, c_mems) # modified
        q_enc, q_mems = self.enc_forward(q_emb, q_mask, q_mems)
        
        att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        # att = self.att(c_enc.transpose(1, 2), q_enc.transpose(1, 2), c_mask, q_mask).transpose(1, 2)
        out = self.att_resize(att.transpose(1, 2)).transpose(1, 2)
#         print("a_mems size", a_mems[0].size())
#         print("out", out.size())
        out1, a_mems = self.mod_forward(out, c_mask, mems=a_mems) # modified
        out2, a_mems = self.mod_forward(out1, c_mask, mems=a_mems)  
        out3, a_mems = self.mod_forward(out2, c_mask, mems=a_mems) 
        
        out = self.out(out1, out2, out3, c_mask)
        # out = self.out(out1.transpose(1, 2), out2.transpose(1, 2), out3.transpose(1, 2), c_mask)

        return out

    def _update_mems(self, hids, mems, q_len, m_len):
        # does not deal with None
        if mems is None: return None

        # mems is not None
#         print("len(hids)",len("hids"))
#         print("len(mems)", len("mems"))
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = m_len + max(0, q_len - 0)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                # hids: batch_size, q_len, hidden_size*2
                cat = torch.cat([mems[i], hids[i].permute(1,0,2)], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

            return new_mems
        
    def _init_mems(self, num_layers):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(num_layers):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None
