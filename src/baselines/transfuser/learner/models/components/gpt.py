import math

import torch
from torch import nn
from torch.nn import functional as F


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, n_embd, n_head, block_exp, n_layer,
                 img_vert_anchors, img_horz_anchors,
                 lidar_vert_anchors, lidar_horz_anchors,
                 seq_len,
                 embd_pdrop, attn_pdrop, resid_pdrop, config, use_velocity=True):
        super().__init__()
        self.n_embd = n_embd
        # We currently only support seq len 1
        self.seq_len = 1

        self.img_vert_anchors = img_vert_anchors
        self.img_horz_anchors = img_horz_anchors
        self.lidar_vert_anchors = lidar_vert_anchors
        self.lidar_horz_anchors = lidar_horz_anchors
        self.config = config

        # positional embedding parameter (learnable), image + lidar
        self.pos_emb = nn.Parameter(torch.zeros(1,
                                                self.seq_len * img_vert_anchors * img_horz_anchors + self.seq_len * lidar_vert_anchors * lidar_horz_anchors,
                                                n_embd))

        # velocity embedding
        self.use_velocity = use_velocity
        if use_velocity:
            self.vel_emb = nn.Linear(self.seq_len, n_embd)

        self.drop = nn.Dropout(embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd, n_head,
                                            block_exp, attn_pdrop, resid_pdrop) for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.block_size = self.seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=self.config.gpt_linear_layer_init_mean,
                                       std=self.config.gpt_linear_layer_init_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor, velocity):
        """
        Args:
            image_tensor (tensor): B*4*seq_len, C, H, W
            lidar_tensor (tensor): B*seq_len, C, H, W
            velocity (tensor): ego-velocity
        """

        bz = lidar_tensor.shape[0]
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        assert self.seq_len == 1
        image_tensor = image_tensor.view(bz, self.seq_len, -1, img_h, img_w).permute(0, 1, 3, 4, 2).contiguous().view(
            bz, -1, self.n_embd)
        lidar_tensor = lidar_tensor.view(bz, self.seq_len, -1, lidar_h, lidar_w).permute(0, 1, 3, 4,
                                                                                         2).contiguous().view(bz, -1,
                                                                                                              self.n_embd)

        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1)

        # project velocity to n_embed
        if self.use_velocity:
            velocity_embeddings = self.vel_emb(velocity)  # (B, C)
            # add (learnable) positional embedding and velocity embedding for all tokens
            x = self.drop(self.pos_emb + token_embeddings + velocity_embeddings.unsqueeze(1))  # (B, an * T, C)
        else:
            x = self.drop(self.pos_emb + token_embeddings)
        x = self.blocks(x)  # (B, an * T, C)
        x = self.ln_f(x)  # (B, an * T, C)

        x = x.view(bz,
                   self.seq_len * self.img_vert_anchors * self.img_horz_anchors + self.seq_len * self.lidar_vert_anchors * self.lidar_horz_anchors,
                   self.n_embd)

        image_tensor_out = x[:, :self.seq_len * self.img_vert_anchors * self.img_horz_anchors, :].contiguous().view(
            bz * self.seq_len, -1, img_h, img_w)
        lidar_tensor_out = x[:, self.seq_len * self.img_vert_anchors * self.img_horz_anchors:, :].contiguous().view(
            bz * self.seq_len, -1, lidar_h, lidar_w)

        return image_tensor_out, lidar_tensor_out


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, block_exp, attn_pdrop, resid_pdrop):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, block_exp * n_embd),
            nn.ReLU(True),  # changed from GELU
            nn.Linear(block_exp * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        return x
