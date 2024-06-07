import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops.misc import MLP
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional, Tuple

class MaskedSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, input, mask=None):
        for module in self.modules_list:
            input = module(input, mask) if mask is not None else module(input)
        return input

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        representation_size: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(representation_size)
        self.self_attention = nn.MultiheadAttention(representation_size, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(representation_size)
        self.mlp = MLPBlock(representation_size, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, representation_size) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, key_padding_mask=mask, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        representation_size: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                representation_size,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = MaskedSequential(layers)
        self.ln = norm_layer(representation_size)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, representation_size) got {input.shape}")
        return self.ln(self.layers(self.dropout(input), mask))

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = F.normalize(x, p=2, dim=-1)
        x = self.last_layer(x)
        return x

class UniViT(nn.Module):
    def __init__(
        self,
        max_height_size: int,
        max_width_size: int,
        max_time_size: int,
        max_slice_size: int,
        num_channels: int,
        patch_size: int,
        representation_size: int,
        num_layers: int,
        num_heads: int,
        projection_size: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        mask_prob: float = 0.15,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        clsExtraHead: bool = False,
    ):
        super().__init__()
        torch._assert(max_height_size % patch_size == 0, "Input height indivisible by patch size!")
        torch._assert(max_width_size % patch_size == 0, "Input width indivisible by patch size!")
        self.image_height = max_height_size
        self.image_width = max_width_size
        self.image_slice = max_slice_size
        self.image_time = max_time_size
        self.image_channels = num_channels
        self.patch_size = patch_size
        self.representation_size = representation_size
        self.projection_size = projection_size
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.mask_prob = mask_prob
        self.norm_layer = norm_layer
        self.conv_proj = nn.Conv2d(
            in_channels=self.image_channels, out_channels=representation_size, kernel_size=patch_size, stride=patch_size
        )
        self.masked_embed = nn.Parameter(torch.zeros(1, 1, representation_size))
        
        self.slice_embedding = nn.Embedding(self.image_slice, representation_size)
        self.time_embedding = nn.Embedding(self.image_time, representation_size)
        self.height_embedding = nn.Linear(1, representation_size)
        self.width_embedding = nn.Linear(1, representation_size)

        seq_length = (self.image_height // patch_size) * (self.image_width // patch_size) * self.image_slice * self.image_time

        # Add class tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, representation_size))
        seq_length += 1
        self.time_class_tokens = nn.Parameter(torch.zeros(1, self.image_time, representation_size))
        seq_length += self.image_time
        self.slice_class_tokens = nn.Parameter(torch.zeros(1, self.image_slice, representation_size))
        seq_length += self.image_slice

        self.encoder = Encoder(
            num_layers,
            num_heads,
            representation_size,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)
            
        self.cls_head = ProjectionHead(representation_size, projection_size)
        self.embed_head = ProjectionHead(representation_size, projection_size)
        self.swap_prob = 0.2
        if clsExtraHead:
            self.cls_extra_head = nn.Sequential(nn.Linear(representation_size, representation_size), nn.ReLU(), nn.Linear(representation_size, representation_size))

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, c, h, w = x.shape
        p = self.patch_size
        torch._assert(t == self.image_time, f"Wrong image time dimension! Expected {self.image_time} but got {t}!")
        torch._assert(s == self.image_slice, f"Wrong image slice dimension! Expected {self.image_slice} but got {s}!")
        torch._assert(c == self.image_channels, f"Wrong number of image channels! Expected {self.image_channels} but got {c}!")
        torch._assert(h == self.image_height, f"Wrong image height! Expected {self.image_height} but got {h}!")
        torch._assert(w == self.image_width, f"Wrong image width! Expected {self.image_width} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (b, t, s, c, h, w) -> (b, representation_size, t * s * n_h * n_w)
        x = x.reshape(b * t * s, c, h, w)
        x = self.conv_proj(x)
        x = x.view(b, t, s, self.representation_size, n_h, n_w).permute(0, 3, 1, 2, 4, 5).reshape(b, self.representation_size, t * s * n_h * n_w)

        # (n, representation_size, (seq_len)) -> (n, (seq_len), representation_size)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def _prepare_sequence(self, x: torch.Tensor, dimensions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, seq_len, _ = x.shape
        patch_height = self.image_height // self.patch_size
        patch_width = self.image_width // self.patch_size
        
        # Generate mask based on dimensions
        mask = torch.zeros((bs, self.image_time, self.image_slice, patch_height, patch_width), dtype=torch.bool, device=x.device)
        time_cls_mask = torch.zeros((bs, self.image_time), dtype=torch.bool)
        slice_cls_mask = torch.zeros((bs, self.image_slice), dtype=torch.bool)
        for i in range(bs):
            mask[i, dimensions[i,0]:, :, :, :] = True
            mask[i, :, dimensions[i,1]:, :, :] = True
            mask[i, :, :, math.ceil(dimensions[i,2] / self.patch_size):, :] = True
            mask[i, :, :, :, math.ceil(dimensions[i,3] / self.patch_size):] = True
            time_cls_mask[i, dimensions[i,0]:] = True
            slice_cls_mask[i, dimensions[i,1]:] = True
        mask = mask.reshape(bs, seq_len)
        
        # Add time and slice embeddings
        time_indices = torch.arange(self.image_time, device=x.device)
        time_emb = self.time_embedding(time_indices)
        time_emb = time_emb.repeat(bs, 1, self.image_slice, patch_height, patch_width, 1).reshape(bs, seq_len, -1)
        
        slice_indices = torch.arange(self.image_slice, device=x.device)
        slice_emb = self.slice_embedding(slice_indices)
        slice_emb = slice_emb.repeat(bs, self.image_time, 1, patch_height, patch_width, 1).reshape(bs, seq_len, -1)
        
        # Add relative height and width embeddings
        h_positions = torch.arange(patch_height, device=x.device).repeat(bs, 1)
        h_positions = h_positions / (torch.ceil(dimensions[:,2] / self.patch_size).unsqueeze(1))
        h_emb = self.height_embedding(h_positions.unsqueeze(-1))
        h_emb = h_emb.repeat(1, self.image_time, self.image_slice, 1, patch_width, 1).reshape(bs, seq_len, -1)
        
        w_positions = torch.arange(patch_width, device=x.device).repeat(bs, 1)
        w_positions = w_positions / (torch.ceil(dimensions[:,3] / self.patch_size).unsqueeze(1))
        w_emb = self.width_embedding(w_positions.unsqueeze(-1))
        w_emb = w_emb.repeat(1, self.image_time, self.image_slice, patch_height, 1, 1).reshape(bs, seq_len, -1)

        # Add positional embeddings to the input and return
        pos_emb = time_emb + slice_emb + h_emb + w_emb
        x = x + pos_emb
        return x, mask, time_cls_mask, slice_cls_mask
    
    def _prepare_sequence_posMask(self, x: torch.Tensor, dimensions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, seq_len, _ = x.shape
        patch_height = self.image_height // self.patch_size
        patch_width = self.image_width // self.patch_size
        
        # Generate mask based on dimensions
        mask = torch.zeros((bs, self.image_time, self.image_slice, patch_height, patch_width), dtype=torch.bool, device=x.device)
        for i in range(bs):
            mask[i, dimensions[i,0]:, :, :, :] = True
            mask[i, :, dimensions[i,1]:, :, :] = True
            mask[i, :, :, math.ceil(dimensions[i,2] / self.patch_size):, :] = True
            mask[i, :, :, :, math.ceil(dimensions[i,3] / self.patch_size):] = True
        mask = mask.reshape(bs, seq_len)
        
        # Add time and slice embeddings
        time_indices = torch.arange(self.image_time, device=x.device)
        time_emb = self.time_embedding(time_indices)
        time_emb = time_emb.repeat(bs, 1, self.image_slice, patch_height, patch_width, 1).reshape(bs, seq_len, -1)
        
        slice_indices = torch.arange(self.image_slice, device=x.device)
        slice_emb = self.slice_embedding(slice_indices)
        slice_emb = slice_emb.repeat(bs, self.image_time, 1, patch_height, patch_width, 1).reshape(bs, seq_len, -1)
        
        # Add relative height and width embeddings
        h_positions = torch.arange(patch_height, device=x.device).repeat(bs, 1)
        h_positions = h_positions / (torch.ceil(dimensions[:,2] / self.patch_size).unsqueeze(1))
        h_emb = self.height_embedding(h_positions.unsqueeze(-1))
        h_emb = h_emb.repeat(1, self.image_time, self.image_slice, 1, patch_width, 1).reshape(bs, seq_len, -1)
        
        w_positions = torch.arange(patch_width, device=x.device).repeat(bs, 1)
        w_positions = w_positions / (torch.ceil(dimensions[:,3] / self.patch_size).unsqueeze(1))
        w_emb = self.width_embedding(w_positions.unsqueeze(-1))
        w_emb = w_emb.repeat(1, self.image_time, self.image_slice, patch_height, 1, 1).reshape(bs, seq_len, -1)

        # Add positional embeddings to the input and return
        pos_emb = time_emb + slice_emb + h_emb + w_emb
        
        pos_mask = torch.rand_like(mask, dtype=torch.float) < self.mask_prob
        pos_emb = pos_emb * (1 - pos_mask.reshape(bs, seq_len, 1))
        
        x = x + pos_emb
        return x, mask, pos_mask
    
    def _mask_sequence(self, x: torch.Tensor, dim: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        mask = torch.rand((bs, self.seq_length), dtype=torch.float, device=x.device) < self.mask_prob
        return mask
    
    def forward(self, x: torch.Tensor, dimensions: torch.Tensor, seqMask: bool = False, train: bool = False, extraCls: bool = False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        if seqMask:
            mask = self._mask_sequence(x, dimensions)
        else:
            mask = torch.zeros((n, self.seq_length), dtype=torch.bool, device=x.device)
            
        x = (mask.float() * self.masked_embed) + ((~mask).float() * x)
        x, orig_mask, time_mask, slice_mask = self._prepare_sequence(x, dimensions)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        batch_class_mask = torch.zeros(n, 1).bool().to(x.device)
        batch_time_class_tokens = self.time_class_tokens.expand(n, -1, -1)
        batch_slice_class_tokens = self.slice_class_tokens.expand(n, -1, -1)
        x = torch.cat([batch_class_token, batch_time_class_tokens, batch_slice_class_tokens, x], dim=1)
        encoder_mask = torch.cat([batch_class_mask, time_mask, slice_mask, orig_mask], dim=1)

        x = self.encoder(x, encoder_mask)
        
        if train:
            cls, x = x[:, 0], x[:, 1:]
            if extraCls:
                return self.cls_head(cls), self.embed_head(x), self.cls_extra_head(x), x, mask, orig_mask, time_mask, slice_mask
            
            return self.cls_head(cls), self.embed_head(x), mask, orig_mask, time_mask, slice_mask
        
        return x
    
    def embed(self, x: torch.Tensor, dimensions: torch.Tensor, train: bool = False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        x, mask, time_mask, slice_mask = self._prepare_sequence(x, dimensions)
        
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        batch_class_mask = torch.zeros(n, 1).bool().to(x.device)
        x = torch.cat([batch_class_token, x], dim=1)
        mask = torch.cat([batch_class_mask, time_mask, slice_mask, mask], dim=1)

        x = self.encoder(x, mask)
        cls = x[:, 0]
        
        if train:
            return self.cls_head(cls)
        
        return cls