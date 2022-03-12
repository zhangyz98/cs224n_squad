import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
from util import myprint, get_available_devices

B, T, C = 2, 5, 12
nh = 4
mask_in = torch.zeros([B, T])
mask_in[0, :3] = 1
mask_in[1, :4] = 1
myprint("input mask", mask_in)
mask = mask_in.view(B, 1, 1, T)
myprint("reshaped mask", mask)

x = torch.rand(B, nh, T, T)
myprint("input x", x)

x = x.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
myprint("masked x", x)

x = F.softmax(x, dim=-1)
myprint("softmaxed x", x)

v = torch.rand(B, nh, T, C // nh)
myprint("v", v)

y = x @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
myprint("x @ v", y)
y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
myprint("reshaped y", y)

softmax_inp = torch.rand(B, T) * 100
myprint('softmax_inp', softmax_inp)
out = masked_softmax(softmax_inp, mask_in, log_softmax=False)
myprint('softmax_out', out)