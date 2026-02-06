import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        # 空表，行代表编码位置（时序位置），列代表度量维度
        pe.require_grad = False
        # 位置embedding是固定的，无需计算梯度

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, win_size, t_model, dropout=0.0):
        super(DataEmbedding_inverted, self).__init__()

        self.value_embedding = nn.Linear(win_size, t_model)
        # 不用conv吗
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbedding_inverted_conv(nn.Module):
    def __init__(self, win_size, t_model, dropout=0.0):
        super(DataEmbedding_inverted_conv, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.value_embedding = nn.Conv1d(in_channels=win_size, out_channels=t_model,
                              kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.position_embedding = PositionalEmbedding(d_model=t_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)
        x = x.permute(0, 2, 1)
        # 一维卷积在最后一个维度上扫，但是通道数对应倒数第二个维度
        return self.dropout(x)

class DataEmbedding_inverted_RNN(nn.Module):
    def __init__(self, t_model, c_in, d_model, dropout=0.0):
        super(DataEmbedding_inverted_RNN, self).__init__()
        self.t_model = t_model
        self.encoder = nn.RNN(input_size=c_in, hidden_size=d_model, num_layers=8, bias=True, batch_first=True, dropout=0.1)
        self.decoder = nn.RNN(input_size=d_model, hidden_size=c_in, num_layers=2, bias=True, batch_first=True, dropout=0.1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        _, state = self.encoder(x)
        state = state.permute(1,0,2).repeat(1, self.t_model//8, 1)
        x, _ = self.decoder(state)
        x = x.permute(0, 2, 1)

        return self.dropout(x)

class Reshuffling():
    def __init__(self, delays):
        super(Reshuffling, self).__init__()
        self.delays = delays
    def forward(self, x):
        x_t = x.shape[2]
        inter_win_num = x_t // self.delays
        x = x.reshape(x.shape[0], x.shape[1], inter_win_num, self.delays)
        x = x[:, :, torch.randperm(x.size(2)), :]
        x = x.reshape(x.shape[0], x.shape[1], x_t)
        return x
# 打乱前后学习出的因果矩阵尽可能相似