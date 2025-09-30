import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None]

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim=64, scale=10.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, t):
        t = t[:, None]
        x_proj = 2 * torch.pi * t * self.W[None, :]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class AttentionCondEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, cond):
        # x:    [B, C]
        # cond: [B, N, C]

        # Normalize for cosine similarity
        x_norm = F.normalize(x, dim=-1)      # [B, C]
        cond_norm = F.normalize(cond, dim=-1)  # [B, N, C]

        # Similarity metrics
        cos_sim = torch.sum(cond_norm * x_norm[:, None, :], dim=-1)   # [B, N]
        euclidean = torch.norm(cond - x[:, None, :], dim=-1)          # [B, N]

        # Find most similar index per sample based on Euclidean distance
        best_idx = torch.argmin(euclidean, dim=1)  # [B]
        
        # Gather best neighbor for each sample
        B = cond.shape[0]
        best_cond = cond[torch.arange(B), best_idx]         # [B, C]
        best_cos_sim = cos_sim[torch.arange(B), best_idx]     # [B]
        best_euclidean = euclidean[torch.arange(B), best_idx] # [B]

        # Concatenate [C] + [1] + [1] -> [C+2]
        cond_feat = torch.cat([
            best_cond, best_cos_sim.unsqueeze(-1), best_euclidean.unsqueeze(-1)
        ], dim=-1)  # [B, C+2]

        return cond_feat

class ScoreModel(nn.Module):
    def __init__(self, cond_num, input_dim, marginal_prob_std, embed_dim=256, channel_num=16, hidden_dim=256):
        super().__init__()
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        self.cond_encoder = AttentionCondEncoder()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dense1 = nn.Linear(embed_dim, hidden_dim)
        self.film1 = nn.Linear(input_dim+2, 2 * hidden_dim)
        self.lnorm1 = nn.LayerNorm(hidden_dim)

        self.conv2 = nn.Conv1d(in_channels=1, out_channels=channel_num, kernel_size=5, stride=2, padding=2, bias=False)
        self.dense2 = Dense(embed_dim, channel_num)
        self.film2 = nn.Linear(input_dim + 2, 2 * channel_num)
        self.gnorm2 = nn.GroupNorm(4, channel_num)

        self.conv3 = nn.Conv1d(in_channels=channel_num, out_channels=channel_num*2, kernel_size=5, stride=2, padding=2, bias=False)
        self.dense3 = Dense(embed_dim, channel_num*2)
        self.film3 = nn.Linear(input_dim + 2, 2 * channel_num * 2)
        self.gnorm3 = nn.GroupNorm(8, channel_num * 2)

        self.conv4 = nn.Conv1d(in_channels=channel_num*2, out_channels=channel_num*4, kernel_size=5, stride=2, padding=2, bias=False)
        self.dense4 = Dense(embed_dim, channel_num*4)
        self.film4 = nn.Linear(input_dim + 2, 2 * channel_num * 4)
        self.gnorm4 = nn.GroupNorm(16, channel_num * 4)

        self.tconv4 = nn.ConvTranspose1d(in_channels=channel_num*4, out_channels=channel_num*2, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.tdense4 = Dense(embed_dim, channel_num*2)
        self.tfilm4 = nn.Linear(input_dim + 2, 2 * channel_num * 2)
        self.tgnorm4 = nn.GroupNorm(8, channel_num*2)

        self.tconv3 = nn.ConvTranspose1d(in_channels=channel_num*4, out_channels=channel_num*1, kernel_size=5, stride=2, padding=2, output_padding=1, bias=False)
        self.tdense3 = Dense(embed_dim, channel_num*1)
        self.tfilm3 = nn.Linear(input_dim + 2, 2 * channel_num)
        self.tgnorm3 = nn.GroupNorm(4, channel_num*1)

        self.tconv2 = nn.ConvTranspose1d(in_channels=channel_num*2, out_channels=1, kernel_size=5, stride=2, padding=2, output_padding=1)
        
        self.tfc1 = nn.Linear(hidden_dim + hidden_dim, input_dim)
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std

    def apply_film(self, x, scale_shift):
        scale, shift = scale_shift.chunk(2, dim=-1)
        if x.dim() == 3:
            scale = scale.unsqueeze(-1)
            shift = shift.unsqueeze(-1)
        return x * (1 + scale) + shift

    def forward(self, x, t, cond=None, use_condition=True):
        embed = self.act(self.embed(t))

        if cond is not None and use_condition:
            cond_feat = self.cond_encoder(x, cond)
        else:
            C = x.shape[-1]
            cond_feat = torch.zeros(x.shape[0], C + 2, device=x.device)
    
        # Encoder
        h1 = self.fc1(x)
        h1 += self.dense1(embed)
        h1 = self.lnorm1(h1)
        h1 = self.apply_film(h1, self.film1(cond_feat))
        h1 = self.act(h1)
        
        h1_1 = h1.unsqueeze(1)

        h2 = self.conv2(h1_1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.apply_film(h2, self.film2(cond_feat))
        h2 = self.act(h2)
        
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.apply_film(h3, self.film3(cond_feat))
        h3 = self.act(h3)
        
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.apply_film(h4, self.film4(cond_feat))
        h4 = self.act(h4)

        # Decoder
        h = self.tconv4(h4)
        h += self.tdense4(embed)
        h = self.tgnorm4(h)
        h = self.apply_film(h, self.tfilm4(cond_feat))
        h = self.act(h)
        
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.tdense3(embed)
        h = self.tgnorm3(h)
        h = self.apply_film(h, self.tfilm3(cond_feat))
        h = self.act(h)

        h = self.tconv2(torch.cat([h, h2], dim=1))

        # Output
        h = h.squeeze(1)
        h = self.tfc1(torch.cat([h, h1], dim=1))
        score = h / self.marginal_prob_std(t)[:, None]
        
        return score