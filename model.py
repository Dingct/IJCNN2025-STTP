from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
import os

class PatchEmbedding(nn.Module):
    def __init__(self, d_model=128, patch_len=1, stride=1, his=12):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his
        self.value_embedding = nn.Linear(self.patch_len*1, d_model, bias=False)
        self.tem_embedding = torch.nn.Parameter(torch.zeros(1, 1, his, d_model))

    def forward(self, x):
        # torch.Size([64, 12, 211, 3])
        batch, _, num_nodes, _ = x.size()
        x = x.permute(0, 2, 3, 1)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            x = x.transpose(2, 3).contiguous().view(batch, num_nodes, self.his//self.patch_len, -1)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len//gap, step=self.stride//gap)
            x = x.transpose(2, 3).contiguous().view(batch, num_nodes, self.his//self.patch_len, -1)
            x = F.pad(x, (0, (self.patch_len - self.patch_len//gap)))
        x = self.value_embedding(x) + self.tem_embedding[:, :, :x.size(2), :]
        return x 

class MemAttention(nn.Module):
    def __init__(self, in_dim, seq_length, num_heads = 1):
        super(MemAttention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.num_heads = num_heads
        self.proj = nn.Linear(in_dim, in_dim)
        assert in_dim % num_heads == 0

        self.mkey = nn.Parameter(torch.randn(seq_length, in_dim))
        nn.init.xavier_uniform_(self.mkey)

        self.mvalue = nn.Parameter(torch.randn(seq_length, in_dim))
        nn.init.xavier_uniform_(self.mvalue)

    def forward(self, x):
        B, T, C = x.shape
        query = self.query(x)
        key = self.mkey.unsqueeze(0).repeat(B,1,1)
        value = self.mvalue.unsqueeze(0).repeat(B,1,1)
        num_heads = self.num_heads
        if num_heads > 1:
            query = torch.cat(torch.chunk(query, num_heads, dim = -1), dim = 0)
            key = torch.cat(torch.chunk(key, num_heads, dim = -1), dim = 0)
            value = torch.cat(torch.chunk(value, num_heads, dim = -1), dim = 0)
        d = value.size(-1)
        energy = torch.matmul(query, key.transpose(-1,-2))
        energy = energy / (d ** 0.5)
        score = torch.softmax(energy, dim = -1)
        head_out = torch.matmul(score, value)
        out = torch.cat(torch.chunk(head_out, num_heads, dim = 0), dim = -1)
        return self.proj(out)

class tinyEncoder(nn.Module):
    def __init__(self,module1,norm1,module2=None,norm2=None,param=None,bias=None,dropout=0.1):
        super(tinyEncoder, self).__init__()
        self.module1 = module1
        self.module2 = module2
        self.norm1 = norm1
        self.norm2 = norm2
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.param = param
        self.bias = bias

    def forward(self, h):
        h_1 = h + self.dropout1(self.module1(h))  # [64, 12, 170, 64]
        h_1 = self.norm1(h_1)
        if self.param is not None:
            h_1 = h_1*self.param + self.bias
        h_2 = h_1
        if self.module2 is not None:
            h_2 = h_1 + self.dropout2(self.module2(h_1))
            h_2 = self.norm2(h_2)
        return h_2

class Stampembedding(nn.Module):
    def __init__(self, timestamp, num_nodes, channel, hist_len, pred_len, dropout=0.1):
        super(Stampembedding, self).__init__()

        self.hist_len = hist_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes

        norm1 = nn.LayerNorm(channel,elementwise_affine=False)
        norm2 = nn.LayerNorm(channel,elementwise_affine=False)
        attn = MemAttention(channel, hist_len, num_heads=4)
        ffw = FeedForward(channel,2*channel,channel)

        self.Encoder = nn.Sequential(
            nn.Linear(timestamp, channel),
            tinyEncoder(attn,norm1,module2=ffw,norm2=norm2,dropout=dropout),
        )

    def forward(self, x_mark_enc):

        x_enc_map = self.Encoder(x_mark_enc)
        
        return x_enc_map

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, d_end, dropout_rate=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff) 
        self.dropout = nn.Dropout(dropout_rate) 
        self.linear2 = nn.Linear(d_ff, d_end)  
        

    def forward(self, x):
        x = self.linear1(x) 
        x = F.relu(x)  
        x = self.dropout(x) 
        return self.linear2(x) 

class Attention(nn.Module):
    def __init__(self, in_dim, num_heads = 4):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.num_heads = num_heads
        self.proj = nn.Linear(in_dim, in_dim)
        assert in_dim % num_heads == 0

    def forward(self, x , mask = None):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        num_heads = self.num_heads
        if num_heads > 1:
            query = torch.cat(torch.chunk(query, num_heads, dim = -1), dim = 0)
            key = torch.cat(torch.chunk(key, num_heads, dim = -1), dim = 0)
            value = torch.cat(torch.chunk(value, num_heads, dim = -1), dim = 0)
        d = value.size(-1)
        energy = torch.matmul(query, key.transpose(-1,-2))
        energy = energy / (d ** 0.5)
        score = torch.softmax(energy, dim = -1)
        if mask is not None:
            score = score * mask
        head_out = torch.matmul(score, value)
        out = torch.cat(torch.chunk(head_out, num_heads, dim = 0), dim = -1)
        return self.proj(out)

class RegionalAttention(nn.Module):
    def __init__(self, num_nodes, num_regions, feature_dim, num_heads=4):
        super(RegionalAttention, self).__init__()
        self.num_nodes = num_nodes
        self.num_regions = num_regions
        self.feature_dim = feature_dim

        # assignment matrix
        self.ass = nn.Parameter(torch.empty(num_nodes, num_regions))
        nn.init.xavier_uniform_(self.ass)  # Xavier

        self.ra = Attention(feature_dim,num_heads=num_heads)
        self.na = Attention(feature_dim,num_heads=num_heads)

        self.alpha_mlp = FeedForward(2 * feature_dim, 4 * feature_dim, feature_dim)

    def forward(self, X): # B F N T
        X = X.transpose(1,3)
        # Processing with softmax to obtain the probabilities
        ass = F.softmax(self.ass, dim=-1)  # [N, Regions]

        # Inter-region 
        rf = torch.einsum('btnf,nr->btrf', X, ass)
        rf = self.ra(rf)
        rf = torch.einsum('btrf,nr->btnf', rf,ass)  

        # Inter-node
        # The pattern similarity of the nodes
        simi = F.cosine_similarity(ass.unsqueeze(1), ass.unsqueeze(0),dim=-1)  # [N, N]
        nf = self.na(X,mask = simi.unsqueeze(0).unsqueeze(0))

        con = torch.cat([rf, nf], dim=-1)
        alp = torch.sigmoid(self.alpha_mlp(con))
        f = alp * rf + (1 - alp) * nf 

        return f.transpose(1,3)

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class PositionwiseFeedForward(nn.Module):
    def __init__(self, channels, act_layer, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, (1, 1))
        self.conv2 = nn.Conv2d(channels, channels, (1, 1))
        self.act = act_layer
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class Satten(nn.Module):
    def __init__(self, num_regions, device, d_model, head, num_nodes, seq_length=12, patch_len=6, dropout=0.1):
        "Take in model size and number of heads."
        super(Satten, self).__init__()
        self.len = seq_length // patch_len

        if num_nodes in [170]: # PEMS08
            self.norm1 = nn.LayerNorm([d_model,num_nodes,self.len], elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = nn.LayerNorm([d_model,num_nodes,self.len], elementwise_affine=True, eps=1e-6)
        self.attn = RegionalAttention(
            num_nodes, num_regions, d_model, num_heads=head
        )
        if num_nodes in [170]:
            self.norm2 = nn.LayerNorm([d_model,num_nodes,self.len], elementwise_affine=False, eps=1e-6)
        else:
            self.norm2 = nn.LayerNorm([d_model,num_nodes,self.len], elementwise_affine=True, eps=1e-6)
        self.mlp = PositionwiseFeedForward(d_model, act_layer=nn.GELU())
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(d_model, 6 * d_model, (1,1))
        )
        
    def forward(self, x, x_mark): # B d_model N T
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(x_mark).chunk(6, dim=1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x

class TConv(nn.Module):
    def __init__(self, features=128, layer=4, length=24, dropout=0.1):
        super(TConv, self).__init__()
        layers = []
        kernel_size = int(length / layer + 1)
        for i in range(layer):
            self.conv = nn.Conv2d(features, features, (1, kernel_size))
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            layers += [nn.Sequential(self.conv, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.tcn(x) + x[..., -1].unsqueeze(-1)
        return x

class STTP(nn.Module):
    def __init__(self, 
                device,
                input_dim=17,
                channels=64,
                num_nodes=170,
                input_len=12,
                output_len=12,
                num_regions=18,
                dropout=0.1, 
                ):
        super().__init__()
        num_heads = 4

        self.num_nodes = num_nodes
        self.emb_x = PatchEmbedding(d_model=channels, patch_len=1, stride=1, his=input_len)
        self.emb_t = Stampembedding(3, num_nodes, channels, input_len, input_len,dropout=dropout)

        self.tmodule1 = TConv(channels, layer=4, length=input_len,dropout=dropout)
        self.tmodule2 = TConv(channels, layer=4, length=input_len,dropout=dropout)

        self.smodule = Satten(num_regions, device, channels, num_heads, num_nodes, seq_length=input_len ,patch_len= input_len, dropout=dropout)
        
        self.decoder = nn.Conv2d(channels, output_len, kernel_size=(1, 1))
    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])
    
    def forward(self, x):
        x_true = x[...,:1] # B T N 1
        x_tp = x[:,:,0,3:6] # B T 3

        emb_x = self.emb_x(x_true).permute(0,3,1,2) # B f N T 
        emb_t = self.emb_t(x_tp).transpose(1,2).unsqueeze(-2) # B f 1 T 

        x_t = self.tmodule1(emb_x) # B f N 1 
        t_t = self.tmodule2(emb_t).repeat(1,1,self.num_nodes,1) # B f N 1 

        xt_ts = self.smodule(x_t,t_t) # B f N 1 

        pre = self.decoder(xt_ts)

        return pre
