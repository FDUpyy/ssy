import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding, DataEmbedding_inverted, DataEmbedding_inverted_conv, DataEmbedding_inverted_RNN

class EncoderLayer(nn.Module):
    def __init__(self, attention, win_size, enc_in, d_model, t_model, delays=3, t_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()

        t_ff = t_ff or 4 * t_model

        self.attention = attention

        self.revert = nn.Linear(d_model, enc_in)
        #self.revert = nn.Conv1d(in_channels=t_model, out_channels=win_size, kernel_size=1)
        #self.revert_c = nn.Conv1d(in_channels=d_model, out_channels=enc_in, kernel_size=1)
        #self.revert_t = nn.Conv1d(in_channels=t_model, out_channels=win_size, kernel_size=1)

        self.conv1 = nn.Conv1d(in_channels=win_size, out_channels=t_model, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=t_model, out_channels=t_ff, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=t_ff, out_channels=t_model, kernel_size=1)

        self.norm1 = nn.LayerNorm(t_model)
        self.norm2 = nn.LayerNorm(t_model)

        self.delays = delays

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_h, x_l, attn_mask=None):

        global new_x_h, series, prior, sigma
        t_size = x_l.shape[2]
        causal_attn_delay = []
        x_l_add = torch.rand(x_l.shape[0], x_l.shape[1], t_size).float().cuda()
        for delay in range(self.delays):
            delay_0 = torch.zeros(x_l.shape[0], x_l.shape[1]).float().unsqueeze(2).cuda()

            x_l = torch.cat((delay_0, x_l), dim=2)

            new_x_h, new_x_l, series, prior, sigma, causal_attn = self.attention(
                x_h, x_h, x_h,
                x_l[:, :, delay + 1:], x_l[:, :, 1:t_size+1], x_l[:, :, 1:t_size+1],
                attn_mask
            )
            x_l_add = x_l_add + new_x_l
            causal_attn_delay.append(causal_attn)

        causal = causal_attn_delay[0]
        """
        causal_num = torch.zeros(causal_attn_delay[0].shape[0], causal_attn_delay[0].shape[1], causal_attn_delay[0].shape[2], causal_attn_delay[0].shape[3]).float().cuda()
        for causal in causal_attn_delay:
            causal_num = causal_num + causal
        causal = causal_num
        """
        # x_l_add = self.dropout(self.activation(self.conv1(self.dropout(self.revert(x_l_add))))).permute(0, 2, 1)
        # x_h = self.dropout(self.activation(self.conv3(self.dropout(self.conv2(x_h.permute(0, 2, 1)).transpose(1, 2))).transpose(1, 2)))
        # x_l = x_l[:, :, self.delays:]
        # x_l = self.dropout(self.activation(self.conv3(self.dropout(self.conv2(x_l.permute(0, 2, 1)).transpose(1, 2))).transpose(1, 2)))

        x = x_l[:, :, self.delays:]
        #x = self.dropout(self.revert(x.transpose(-1, 1)).transpose(-1, 1))  # x=(256,38,100)
        new_x_h = self.dropout(self.activation(self.conv1(self.dropout(self.revert(new_x_h))))).permute(0, 2, 1)
        #new_x_h = self.dropout(self.revert_c(new_x_h.transpose(-1, 1))) #new_x_h=(256,38,100)
        #x_l_add = self.dropout(self.revert_t(x_l_add.transpose(-1, 1)).transpose(-1, 1)) #new_x_h=(256,38,100)
        new_x = new_x_h + x_l_add

        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv2(y.transpose(-1, 1))))
        y = self.dropout(self.conv3(y).transpose(-1, 1))

        return self.norm2(x+y), series, prior, sigma, causal
        # series = (B H L S)
        # prior = (B H L L)
        # sigma = (B H L L)

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer, delays, dropout=0.1, activation="relu"):
        # 论文单L层Encoder层的一些神经网络固定参数：
        # attn_layers：EncoderLayer类，在forward里循环调用L次这个类的forward函数
        # norm_layer：LayerNorm层
        super(Encoder, self).__init__()
        self.delays = delays
        self.attn_layers = nn.ModuleList(attn_layers)
        # nn.ModuleList()：将多个子模块组织在一起，类似nn.Sequential()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.norm = norm_layer

    def forward(self, x_h, x_l, attn_mask=None):
        # x_h [B, L, D_512]
        # x_l [B, D, L_512]
        series_list = []
        prior_list = []
        sigma_list = []
        causal_list = []

        for attn_layer in self.attn_layers:
            x, series, prior, sigma, causal = attn_layer(x_h, x_l, attn_mask)
            # causal_delay因果矩阵：[delay B H D D]的delay维张量
            # L个编码器堆叠（串联），不需要取平均
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            causal_list.append(causal)

        #causal = causal_list[-1]

        causal_num = torch.zeros(causal_list[0].shape[0], causal_list[0].shape[1],
                                 causal_list[0].shape[2], causal_list[0].shape[3]).float().cuda()
        for causal in causal_list:
            causal_num = causal_num + causal
        causal = causal_num / len(causal_list)


        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list, causal
        # x_h = (B L d_model)
        # x_l = (B D d_model), d_model=t_model
        # series_list = (self.attn_layers B H L S)
        # prior_list = (self.attn_layers B H L L)
        # sigma_list = (self.attn_layers B H L L)
        # causal = (B H D D)

"""
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, t_model, delays=3, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model

        self.attention = attention

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=t_model, out_channels=d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=d_ff, out_channels=t_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(t_model)
        self.norm4 = nn.LayerNorm(t_model)

        self.delays = delays

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_h, x_l, attn_mask=None):

        global new_x_h, series, prior, sigma
        t_size = x_l.shape[2]
        causal_attn_delay = []
        x_l_add = torch.zeros(x_l.shape[0], x_l.shape[1], t_size).float().cuda()
        for delay in range(self.delays):
            delay_0 = torch.zeros(x_l.shape[0], x_l.shape[1]).float().unsqueeze(2).cuda()

            x_l = torch.cat((delay_0, x_l), dim=2)

            new_x_h, new_x_l, series, prior, sigma, causal_attn = self.attention(
                x_h, x_h, x_h,
                x_l[:, :, delay + 1:], x_l[:, :, 1:t_size+1], x_l[:, :, 1:t_size+1],
                attn_mask
            )
            x_l_add = x_l_add + new_x_l
            causal_attn_delay.append(causal_attn)

        causal = causal_attn_delay[0]

        x_h = x_h + self.dropout(new_x_h)
        y_h = x_h = self.norm1(x_h)
        y_h = self.dropout(self.activation(self.conv1(y_h.transpose(-1, 1))))
        y_h = self.dropout(self.conv2(y_h).transpose(-1, 1))

        x_l = x_l[:, :, self.delays:] + self.dropout(x_l_add)
        y_l = x_l = self.norm3(x_l)
        y_l = self.dropout(self.activation(self.conv3(y_l.transpose(-1, 1))))
        y_l = self.dropout(self.conv4(y_l).transpose(-1, 1))

        return self.norm2(x_h+y_h), self.norm4(x_l+y_l), series, prior, sigma, causal
        # series = (B H L S)
        # prior = (B H L L)
        # sigma = (B H L L)

class Encoder(nn.Module):
    def __init__(self, attn_layers, win_size, enc_in, d_model, norm_layer, delays, dropout=0.1, activation="relu"):
        # 论文单L层Encoder层的一些神经网络固定参数：
        # attn_layers：EncoderLayer类，在forward里循环调用L次这个类的forward函数
        # norm_layer：LayerNorm层
        super(Encoder, self).__init__()
        self.delays = delays
        self.attn_layers = nn.ModuleList(attn_layers)
        # nn.ModuleList()：将多个子模块组织在一起，类似nn.Sequential()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.revert = nn.Linear(d_model, win_size)
        self.conv = nn.Conv1d(in_channels=enc_in, out_channels=d_model, kernel_size=1)
        self.norm = norm_layer

    def forward(self, x_h, x_l, attn_mask=None):
        # x_h [B, L, D_512]
        # x_l [B, D, L_512]
        series_list = []
        prior_list = []
        sigma_list = []
        causal_list = []

        for attn_layer in self.attn_layers:
            x_h, x_l, series, prior, sigma, causal = attn_layer(x_h, x_l, attn_mask)
            # causal_delay因果矩阵：[delay B H D D]的delay维张量
            # L个编码器堆叠（串联），不需要取平均
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            causal_list.append(causal)

        causal = causal_list[-1]
        x_l = self.dropout(self.activation(self.revert(x_l)))
        x_l = self.dropout(self.activation(self.conv(x_l))).permute(0, 2, 1)

        if self.norm is not None:
            x = self.norm(x_h + x_l)

        return x, series_list, prior_list, sigma_list, causal
        # x_h = (B L d_model)
        # x_l = (B D d_model), d_model=t_model
        # series_list = (self.attn_layers B H L S)
        # prior_list = (self.attn_layers B H L L)
        # sigma_list = (self.attn_layers B H L L)
        # causal = (B H D D)
"""
class AnomalyTransformer(nn.Module):
    def __init__(self, win_size, enc_in, t_out, delays, t_model=512, d_model=512, n_heads=8, e_layers=3, t_ff=2048,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()

        self.t_out = t_out
        self.output_attention = output_attention
        # Encoding
        self.embedding_temporal = DataEmbedding(enc_in, d_model, dropout)
        self.embedding_causal = DataEmbedding_inverted_conv(win_size, t_model, dropout)
        #self.embedding_causal = DataEmbedding_inverted_RNN(t_model, enc_in, d_model, dropout)


        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(win_size, True, attention_dropout=dropout, output_attention=output_attention),
                        d_model, t_model, n_heads),
                    win_size,
                    enc_in,
                    d_model,
                    t_model,
                    delays,
                    t_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],

            norm_layer=torch.nn.LayerNorm(t_model),
            delays=delays
        )
        self.projection = nn.Linear(t_model, t_out)
        #self.projection_enc = nn.RNN(input_size=enc_in, hidden_size=d_model, num_layers=5, bias=True, batch_first=True, dropout=0.1)
        #self.projection_dec = nn.RNN(input_size=d_model, hidden_size=enc_in, num_layers=2, bias=True, batch_first=True, dropout=0.1)
        # 怎么改？不用改应该

    def forward(self, x_h, x_l):
        enc_out_h = self.embedding_temporal(x_h)
        # (256,100,512)
        enc_out_l = self.embedding_causal(x_l)
        # enc_out_l = x_l.permute(0, 2, 1)
        # (256,38,512)
        enc_out, series, prior, sigmas, causal = self.encoder(enc_out_h, enc_out_l)
        enc_out = self.projection(enc_out)

        #_, state = self.projection_enc(enc_out.transpose(-1, 1))
        #state = state.permute(1, 0, 2).repeat(1, self.t_out // 5, 1)
        #enc_out, _ = self.projection_dec(state)

        if self.output_attention:
            # self.output_attention=True代表什么？
            return enc_out.transpose(-1, 1), series, prior, sigmas, causal
            # series = (self.attn_layers B H L S)
            # prior = (self.attn_layers B H L L)
            # sigma = (self.attn_layers B H L L)
            # causal = (B H D D)
        else:
            return enc_out.transpose(-1, 1)
            # [B, L, D]
