import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale_h=None, scale_l=None, attention_dropout=0.0, output_attention=False):
        # 论文anomaly attention层的一些神经网络固定参数：输入窗口大小（即输入时序长度）、对角线因果掩码机制（）
        # scale
        # dropout层（什么时候用呢？）
        # output_attention（用于判断是否完整调用了anomalyattention应当具备的输入参数）
        super(AnomalyAttention, self).__init__()
        self.scale_h = scale_h
        self.scale_l = scale_l

        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        # 这里的window_size只用于self.distances()，与forward中的不同，所以不用self声明
        self.distances = torch.zeros((window_size, window_size)).cuda()
        # distance用于高斯先验prior的计算
        # cuda()用于将张量（Tensor）从CPU迁移到GPU上进行计算。这个操作通常是在模型的创建或转换时进行的
        # 而不是在执行某个运算时动态指定
        # 因此，不建议在运行时通过call.cuda()将模型加载到GPU上
        # 而应该优先考虑使用model.to(device)的方法，这样可以在代码中显式指定所需的计算设备,特别是当有多个GPU可用时
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)
                # abs()绝对值函数，两个下标之间的距离

    def forward(self, queries_h, keys_h, values_h, queries_l, keys_l, values_l, sigma, attn_mask):
        # 在代码中上一层AttentionLayer中调用时需要输入的参数，用于论文anomaly attention层的一些计算参数
        B, L, H, E_h = queries_h.shape
        # B（batch_size）、L(len_q，长度)、H（heads_number，注意力头数）、E（d_q=d_k，维度）
        _, S, _, D_h = values_h.shape
        # S（len_k=len_v，长度）、D（d_v，维度）

        scale_h = self.scale_h or 1. / sqrt(E_h)
        # or两边如果是逻辑表达式，则返回true/false
        # or两边如果是变量，则返回第一个true的变量值，如果所有表达式都为false（0、None、“”），则返回最后一个表达式的值

        scores_h = torch.einsum("blhe,bshe->bhls", queries_h, keys_h)
        """
        q = queries_h.transpose(-1, 1).transpose(2, 1).unsqueeze(-1)
        k = keys_h.transpose(-1, 1).transpose(2, 1).unsqueeze(-2)
        scores_h = torch.matmul(q, k).view(B, H, E_h, L * S)
        #scores_h_mask = torch.sum(scores_h, dim=2).view(B, H, L, S)
        scores_h, _ = torch.min(scores_h, dim=2)
        scores_h = scores_h.view(B, H, L, S)
        """
        # blhe是qurries下标，bhls是keys下标，即Q*转置K
        # 当Q和K的长度不一样时，要用加性注意力机制，不能用缩放点积注意力机制，所以这里S=L，后面都是[L,L]
        """
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries_h.device)
            scores_h.masked_fill_(attn_mask.mask, -np.inf)
            # -np.inf表示负无穷大
            # attn_mask.mask表示一个L*L的上三角矩阵为真值，可见TriangularCausalMask()
            # masked_fill_()表示将scores的attn_mask.mask为真值的元素替换为-np.inf
            # 后面经过softmax，负无穷大的部分会变为0
            # causal transformer这部分应该不能这么做，因为不是causal不是对称关系
            # 这里是借鉴的informer，减少计算
            # 原始transformer中的encoder并未使用attn_mask，只使用了padding_mask(时序预测中也用不到，nlp用来统一字符的编码长度)
        """
        attn_h = scale_h * scores_h

        # attention值
        series = self.dropout(torch.softmax(attn_h, dim=-1))
        # softmax(attention)，对最后一维（S）进行非线性映射

        attn_h_mask = attn_h
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries_h.device)
            attn_h_mask.masked_fill_(attn_mask.mask, -np.inf)
        series_mask = self.dropout(torch.softmax(attn_h_mask, dim=-1))

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn_h.shape[-1]
        # shape[-1]代表最后一个维度number，即len_k=len_v
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        # pow()函数：3的sigma次方
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        # unsqueeze(-1) 在sigma最后添加一维
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda()
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))

        V_h = torch.einsum("bhls,bshd->blhd", series_mask, values_h)

        B, E, H, L_l = queries_l.shape
        # E = v_q， L = t_q
        _, D, _, S_l = values_l.shape
        # D = v_v， S = t_v

        scale_l = self.scale_l or 1. / sqrt(L_l)
        scores_l = torch.einsum("behl,bdhl->bhed", queries_l, keys_l)

        attn_l = scale_l * scores_l

        causal = self.dropout(torch.softmax(attn_l, dim=-1))
        V_l = torch.einsum("bhed,bdhs->behs", causal, values_l)

        if self.output_attention:
            return (V_h.contiguous(), V_l.contiguous(), series, prior, sigma, causal)
            # 不太确定什么时候必须用contiguous()
            # V_h.contiguous() = (B L H D_512)
            # series = (B H L S)
            # prior = (B H L L)
            # sigma = (B H L L)
        else:
            return (V_h.contiguous(), V_l.contiguous(), None)
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, t_model, n_heads, d_keys=None,
                 d_values=None, t_keys=None, t_values=None):
        # 论文多头anomaly attention层的一些神经网络固定参数：
        # attention：AnomalyAttention类，在forward里调用这个类的forward函数
        # d_model:embedding后的输入维度、n_heads:头数
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (t_model // n_heads)

        t_keys = t_keys or (t_model // n_heads)
        t_values = t_values or (t_model // n_heads)

        self.norm_h = nn.LayerNorm(d_model)
        # 貌似没用到
        self.norm_l = nn.LayerNorm(t_model)
        # 貌似没用到

        self.inner_attention = attention
        # 建立多头attention层

        self.query_h_projection = nn.Linear(d_model, d_keys * n_heads)
        # d_model表示输入Tensor的最后一维的通道数，d_keys * n_heads表示输出Tensor的最后一维的通道数
        # 用self.query_projection(input)调用nn
        self.key_h_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_h_projection = nn.Linear(d_model, d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model, n_heads)

        self.query_l_projection = nn.Linear(t_model, t_keys * n_heads)
        self.key_l_projection = nn.Linear(t_model, t_keys * n_heads)
        self.value_l_projection = nn.Linear(t_model, t_values * n_heads)

        self.out_h_projection = nn.Linear(d_values * n_heads, d_model)
        self.out_l_projection = nn.Linear(t_values * n_heads, t_model)

        self.n_heads = n_heads

    def forward(self, queries_h, keys_h, values_h, queries_l, keys_l, values_l, attn_mask=None):
        B, L, _ = queries_h.shape
        _, S, _ = keys_h.shape
        H = self.n_heads
        x = queries_h

        B, E, _ = queries_l.shape
        _, D, _ = keys_l.shape

        queries_h = self.query_h_projection(queries_h).view(B, L, H, -1)
        # view()重新调整Tensor形状（B,L,H,E_512/H）
        keys_h = self.key_h_projection(keys_h).view(B, S, H, -1)
        # （B,S,H,E_512/H）
        values_h = self.value_h_projection(values_h).view(B, S, H, -1)
        # （B,S,H,E_512/H）
        sigma = self.sigma_projection(x).view(B, L, H)
        # （B,L,H）

        queries_l = self.query_l_projection(queries_l).view(B, E, H, -1)
        keys_l = self.key_l_projection(keys_l).view(B, D, H, -1)
        values_l = self.value_l_projection(values_l).view(B, D, H, -1)

        out_h, out_l, series, prior, sigma, causal = self.inner_attention(
            queries_h,
            keys_h,
            values_h,
            queries_l,
            keys_l,
            values_l,
            sigma,
            attn_mask
        )

        # 调用AnomalyAttention的forward()
        out_h = out_h.view(B, L, -1)
        # （B,L,D_512）
        out_l = out_l.view(B, E, -1)
        # （B,E,L_512）

        return self.out_h_projection(out_h), self.out_l_projection(out_l), series, prior, sigma, causal
        # self.out_projection(out) = （B,L,d_model）
        # series = (B H L S)
        # prior = (B H L L)
        # sigma = (B H L L)
