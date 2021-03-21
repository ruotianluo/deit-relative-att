
import torch
import torch.nn as nn
from functools import partial


def remove_position_embedding(vit):
    tmp = vit.pos_embed
    del vit.pos_embed
    vit.register_buffer('pos_embed', tmp.data)


# Copy from timm, just for reference
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def att_to_prog_att(att_module):
    att_module.__class__ = ProgressiveAttention


def build_distance_matrix(img_size, patch_size):
    # calculate distance
    rows, cols = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
    
    distance_matrix = torch.zeros(rows, cols, rows, cols)
    for i in range(rows):
        for j in range(cols):
            for ii in range(rows):
                for jj in range(cols):
                    # several way to define it. let start with convolution.
                    distance_matrix[i,j,ii,jj] = max(abs(ii-i), abs(jj-j))
                    # other
                    # mahattan
                    # l2 distance
    distance_matrix = distance_matrix.view(rows*cols, rows*cols)
    # to add the cls_token
    new_distance_matrix = torch.zeros(1+rows*cols, 1+rows*cols)
    # every token is 0 distance to cls_token, and cls_token is zero distance to any other tokens.
    new_distance_matrix[1:,1:] = distance_matrix
    return new_distance_matrix


class ProgressiveAttention(Attention):
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # only change here: progressive
        attn = apply_local_mask(attn)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def apply_local_mask(attn):
    global local_att_mask
    # make sure on the same device
    if local_att_mask.device != attn.device:
        local_att_mask = local_att_mask.to(attn.device)
    attn.add_(local_att_mask)
    return attn


def update_local_mask(epoch, func):
    progressive_att_dist_threshold = func(epoch)
    binary_mask = progressive_att_distance_matrix <= progressive_att_dist_threshold
    global local_att_mask
    local_att_mask = torch.full_like(progressive_att_distance_matrix, float('-inf'), dtype=torch.float32)
    local_att_mask[binary_mask] = 0


def do_surgery(model, attention_surgery, attention_surgery_args):
    if attention_surgery == 'progressive':
        global progressive_att_distance_matrix
        progressive_att_distance_matrix = build_distance_matrix(
            model.patch_embed.img_size,
            model.patch_embed.patch_size,
        )
        for block in model.blocks:
            att_to_prog_att(block.attn)
        attention_surgery_args = eval(attention_surgery_args[1:])
        update_func = attention_surgery_args.get('update_func',
            lambda x: x // 5 # starts from 60 (if 14x14 patches), it become full attention
        )
        update_local_mask(0, update_func)
        return partial(update_local_mask, func=update_func)
    elif attention_surgery == 't5_relative':
        remove_position_embedding(model)
        for block in model.blocks:
            block.attn.__class__ = T5RelativeAttention
            # For simplicity, we don't share rel pos embed across layers.
            block.attn.add_betas(
                model.patch_embed.img_size,
                model.patch_embed.patch_size,
            )
    elif attention_surgery == 'relative':
        remove_position_embedding(model)
        for block in model.blocks:
            block.attn.__class__ = RelativeAttention
            block.attn.add_As(
                model.patch_embed.img_size,
                model.patch_embed.patch_size,
            )


class T5RelativeAttention(Attention):
    """
    Reference: https://arxiv.org/pdf/1910.10683.pdf
    We use a simplified form of position embeddings
where each “embedding” is simply a scalar that is addedto the corresponding logit used
for computing the attention weights. For efficiency, we also share the position embedding
parameters across all layers in our model, though within a given layer each attention head
uses a different learned position embedding. 
    """
    def add_betas(self, img_size, patch_size):
        self.register_buffer('mapping_indices',
            self.create_index_mapping(img_size, patch_size), persistent=False)
        N = self.mapping_indices.shape[0]  # how many patches
        self.beta1 = nn.Parameter(torch.zeros(self.num_heads, 1, 1+N))
        self.beta2 = nn.Parameter(torch.zeros(self.num_heads, N, 1))
        self.beta3 = nn.Parameter(torch.zeros(self.num_heads, self.mapping_indices.max().item()+1))
        # the idea is
        # beta  = 
        # b rearraged(beta)
        # e
        # t
        # a
        # 2
        # b e t a 1

    def get_beta_matrix(self):
        N = self.mapping_indices.shape[0]
        beta_matrix = torch.cat([
            torch.cat([self.beta2, self.beta3[:, self.mapping_indices]], 2),
            self.beta1], dim=1)
        return beta_matrix

    def create_index_mapping(self, img_size, patch_size):
        # calculate distance
        rows, cols = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        
        mapping = {}
        for i in range(-rows+1, rows):
            for j in range(-cols+1, cols):
                mapping[(i,j)] = len(mapping)

        mapping_indices = torch.zeros(rows, cols, rows, cols).long()
        for i in range(rows):
            for j in range(cols):
                for ii in range(rows):
                    for jj in range(cols):
                        # several way to define it. let start with convolution.
                        mapping_indices[i,j,ii,jj] = mapping[(ii-i, jj-j)]
        return mapping_indices.view(rows*cols, rows*cols)
            
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # only change here
        attn += self.get_beta_matrix() # h x N x N

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RelativeAttention(Attention):
    """
    Self-Attention with Relative Position Representations
    """
    def add_As(self, img_size, patch_size):
        self.register_buffer('mapping_indices',
            self.create_index_mapping(img_size, patch_size), persistent=False)
        N = self.mapping_indices.shape[0]  # how many patches
        C = self.qkv.in_features
        self.A_1 = nn.Parameter(torch.zeros(2, self.num_heads, 1, 1+N, C//self.num_heads))
        self.A_2 = nn.Parameter(torch.zeros(2, self.num_heads, N, 1, C//self.num_heads))
        self.A_3 = nn.Parameter(torch.zeros(2, self.num_heads, self.mapping_indices.max().item()+1, C//self.num_heads))

    def get_A_matrix(self):
        # A_matrix[0] is A_k
        # A_matrix[1] is A_v
        N = self.mapping_indices.shape[0]
        A_matrix = torch.cat([
            torch.cat([self.A_2, self.A_3[:, :, self.mapping_indices]], 3),
            self.A_1], dim=2)
        return A_matrix

    def create_index_mapping(self, img_size, patch_size):
        # calculate distance
        rows, cols = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        
        mapping = {}
        for i in range(-rows+1, rows):
            for j in range(-cols+1, cols):
                mapping[(i,j)] = len(mapping)

        mapping_indices = torch.zeros(rows, cols, rows, cols).long()
        for i in range(rows):
            for j in range(cols):
                for ii in range(rows):
                    for jj in range(cols):
                        # several way to define it. let start with convolution.
                        mapping_indices[i,j,ii,jj] = mapping[(ii-i, jj-j)]
        return mapping_indices.view(rows*cols, rows*cols)
            
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # change start
        A = self.get_A_matrix()
        A_k = A[0] # h x N x N x d
        A_v = A[1]

        attn = q @ k.transpose(-2, -1)
        attn = attn + torch.einsum('b h n d, h n m d -> b h n m', q, A_k)
        attn = attn * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x + torch.einsum('b h n m, h n m d -> b h n d', attn, A_v)
        x = x.transpose(1, 2).reshape(B, N, C)
        # change end
        x = self.proj(x)
        x = self.proj_drop(x)
        return x