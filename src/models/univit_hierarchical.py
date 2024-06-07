import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional, Tuple

class MaskedSequential(nn.Module):
    def __init__(self, args):
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
        self.layers = MaskedSequential(layers.values())
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
        num_secondary_layers: int,
        num_heads: int,
        projection_size: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        mask_prob: float = 0.15,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        posTypeHead: bool = False,
        posMaskHead: bool = False,
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

        # Add class tokens
        self.image_class_token = nn.Parameter(torch.zeros(1, 1, representation_size))
        self.slice_class_token = nn.Parameter(torch.zeros(1, 1, representation_size))
        self.time_class_token = nn.Parameter(torch.zeros(1, 1, representation_size))

        self.image_encoder = Encoder(
            num_layers,
            num_heads,
            representation_size,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.slice_encoder = Encoder(
            num_secondary_layers,
            num_heads,
            representation_size,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.time_encoder = Encoder(
            num_secondary_layers,
            num_heads,
            representation_size,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )

        # Init the patchify stem
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)
            
        self.cls_head = ProjectionHead(representation_size, projection_size)
        self.embed_head = ProjectionHead(representation_size, projection_size)
        self.swap_prob = 0.2
        if posTypeHead:
            self.pos_head = nn.Sequential(nn.Linear(representation_size, representation_size), nn.ReLU(), nn.Linear(representation_size, 1))
        elif posMaskHead:
            self.pos_head = nn.Sequential(nn.Linear(representation_size, representation_size), nn.ReLU(), nn.Linear(representation_size, self.image_height + self.image_width + self.image_slice + self.image_time))

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        b, t, s, c, h, w = x.shape
        p = self.patch_size
        torch._assert(t == self.image_time, f"Wrong image time dimension! Expected {self.image_time} but got {t}!")
        torch._assert(s == self.image_slice, f"Wrong image slice dimension! Expected {self.image_slice} but got {s}!")
        torch._assert(c == self.image_channels, f"Wrong number of image channels! Expected {self.image_channels} but got {c}!")
        torch._assert(h == self.image_height, f"Wrong image height! Expected {self.image_height} but got {h}!")
        torch._assert(w == self.image_width, f"Wrong image width! Expected {self.image_width} but got {w}!")

        # (b, t, s, c, h, w) -> (b * t * s, representation_size, n_h * n_w)
        x = x.reshape(b * t * s, c, h, w)
        x = self.conv_proj(x)
        x = x.reshape(b * t * s, self.representation_size, -1)

        # (n, representation_size, (seq_len)) -> (n, (seq_len), representation_size)
        x = x.permute(0, 2, 1)

        return x
    
    def _prepare_sequence(self, x: torch.Tensor, dimensions: torch.Tensor, swapPos: bool = False, adversarialPos = None, adversarialPatch = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, _ = dimensions.shape
        patch_height = self.image_height // self.patch_size
        patch_width = self.image_width // self.patch_size

        # Generate mask based on dimensions
        image_mask = torch.zeros((bs, self.image_time, self.image_slice, patch_height, patch_width), dtype=torch.bool, device=x.device)
        slice_mask = torch.zeros((bs, self.image_time, self.image_slice), dtype=torch.bool, device=x.device)
        time_mask = torch.zeros((bs, self.image_time), dtype=torch.bool, device=x.device)
        for i in range(bs):
            time_mask[i, dimensions[i,0]:] = True
            slice_mask[i, :, dimensions[i,1]:] = True
            image_mask[i, :, :, math.ceil(dimensions[i,2] / self.patch_size):, :] = True
            image_mask[i, :, :, :, math.ceil(dimensions[i,3] / self.patch_size):] = True
        image_mask = image_mask.reshape(bs * self.image_time * self.image_slice, -1)
        slice_mask = slice_mask.reshape(bs * self.image_time, -1)

        # Create Embeddings Per Level
        time_indices = torch.arange(self.image_time, device=x.device)
        time_emb = self.time_embedding(time_indices)
        time_emb = time_emb.repeat(bs, 1, 1)

        slice_indices = torch.arange(self.image_slice, device=x.device)
        slice_emb = self.slice_embedding(slice_indices)
        slice_emb = slice_emb.repeat(bs, self.image_time, 1, 1).view(-1, self.image_slice, self.representation_size)

        h_positions = torch.arange(patch_height, device=x.device).repeat(bs, 1)
        h_positions = h_positions / (torch.ceil(dimensions[:,2] / self.patch_size).unsqueeze(1).to(x.device))
        h_emb = self.height_embedding(h_positions.unsqueeze(-1))
        h_emb = h_emb.repeat(1, self.image_time, self.image_slice, 1, patch_width, 1).reshape(-1, patch_height * patch_width, self.representation_size)

        w_positions = torch.arange(patch_width, device=x.device).repeat(bs, 1)
        w_positions = w_positions / (torch.ceil(dimensions[:,3] / self.patch_size).unsqueeze(1).to(x.device))
        w_emb = self.width_embedding(w_positions.unsqueeze(-1))
        w_emb = w_emb.repeat(1, self.image_time, self.image_slice, patch_height, 1, 1).reshape(-1, patch_height * patch_width, self.representation_size)

        image_emb = h_emb + w_emb

        if swapPos:
            image_seq_len = (self.image_height / self.patch_size) * (self.image_width / self.patch_size)
            num_image_swaps = int(image_seq_len * self.swap_prob)
            for b in range(bs * self.image_time * self.image_slice):
                indices_a = torch.randperm(image_seq_len)[:num_image_swaps]
                indices_b = torch.randperm(image_seq_len)[:num_image_swaps]
                temp = image_emb[b, indices_a].clone()
                image_emb[b, indices_a], image_emb[b, indices_b] = image_emb[b, indices_b], temp
                
            slice_seq_len = self.image_slice
            num_slice_swaps = int(slice_seq_len * self.swap_prob)
            for b in range(bs * self.image_time):
                indices_a = torch.randperm(slice_seq_len)[:num_slice_swaps]
                indices_b = torch.randperm(slice_seq_len)[:num_slice_swaps]
                temp = slice_emb[b, indices_a].clone()
                slice_emb[b, indices_a], slice_emb[b, indices_b] = slice_emb[b, indices_b], temp
                
            time_seq_len = self.image_time
            num_time_swaps = int(time_seq_len * self.swap_prob)
            for b in range(bs):
                indices_a = torch.randperm(time_seq_len)[:num_time_swaps]
                indices_b = torch.randperm(time_seq_len)[:num_time_swaps]
                temp = time_emb[b, indices_a].clone()
                time_emb[b, indices_a], time_emb[b, indices_b] = time_emb[b, indices_b], temp
        elif adversarialPos is not None:
            pos_image_mask = torch.rand_like(image_mask, dtype=torch.float) < self.mask_prob
            image_emb[pos_image_mask] = adversarialPos(image_emb[pos_image_mask])
            pos_slice_mask = torch.rand_like(slice_mask, dtype=torch.float) < self.mask_prob
            slice_emb[pos_slice_mask] = adversarialPos(slice_emb[pos_slice_mask])
            pos_time_mask = torch.rand_like(time_mask, dtype=torch.float) < self.mask_prob
            time_emb[pos_time_mask] = adversarialPos(time_emb[pos_time_mask])
        

        if adversarialPatch is not None:
            patch_mask = torch.rand_like(x, dtype=torch.float) < self.mask_prob
            x[patch_mask] = adversarialPatch(x[patch_mask])
        
        return image_mask, slice_mask, time_mask, image_emb, slice_emb, time_emb
    
    def _prepare_sequence_posMask(self, x: torch.Tensor, dimensions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, _, _ = x.shape
        patch_height = self.image_height // self.patch_size
        patch_width = self.image_width // self.patch_size
        
        # Generate mask based on dimensions
        image_mask = torch.zeros((bs, self.image_time, self.image_slice, patch_height, patch_width), dtype=torch.bool, device=x.device)
        slice_mask = torch.zeros((bs, self.image_time, self.image_slice), dtype=torch.bool, device=x.device)
        time_mask = torch.zeros((bs, self.image_time), dtype=torch.bool, device=x.device)
        for i in range(bs):
            time_mask[i, dimensions[i,0]:] = True
            slice_mask[i, :, dimensions[i,1]:] = True
            image_mask[i, :, :, math.ceil(dimensions[i,2] / self.patch_size):, :] = True
            image_mask[i, :, :, :, math.ceil(dimensions[i,3] / self.patch_size):] = True
        image_mask = image_mask.reshape(bs * self.image_time * self.image_slice, -1)
        slice_mask = slice_mask.reshape(bs * self.image_time, -1)

        # Create Embeddings Per Level
        time_indices = torch.arange(self.image_time, device=x.device)
        time_emb = self.time_embedding(time_indices)
        time_emb = time_emb.repeat(bs, 1, 1)

        slice_indices = torch.arange(self.image_slice, device=x.device)
        slice_emb = self.slice_embedding(slice_indices)
        slice_emb = slice_emb.repeat(bs, self.image_time, 1, 1).view(-1, self.image_slice, self.representation_size)

        h_positions = torch.arange(patch_height, device=x.device).repeat(bs, 1)
        h_positions = h_positions / (torch.ceil(dimensions[:,2] / self.patch_size).unsqueeze(1).to(x.device))
        h_emb = self.height_embedding(h_positions.unsqueeze(-1))
        h_emb = h_emb.repeat(1, self.image_time, self.image_slice, 1, patch_width, 1).reshape(-1, patch_height * patch_width, self.representation_size)

        w_positions = torch.arange(patch_width, device=x.device).repeat(bs, 1)
        w_positions = w_positions / (torch.ceil(dimensions[:,3] / self.patch_size).unsqueeze(1).to(x.device))
        w_emb = self.width_embedding(w_positions.unsqueeze(-1))
        w_emb = w_emb.repeat(1, self.image_time, self.image_slice, patch_height, 1, 1).reshape(-1, patch_height * patch_width, self.representation_size)

        image_emb = h_emb + w_emb
        
        pos_image_mask = torch.rand_like(image_mask, dtype=torch.float) < self.mask_prob
        image_emb = image_emb * (1 - pos_image_mask.unsqueeze(-1))
        pos_slice_mask = torch.rand_like(slice_mask, dtype=torch.float) < self.mask_prob
        slice_emb = slice_emb * (1 - pos_slice_mask.unsqueeze(-1))
        pos_time_mask = torch.rand_like(time_mask, dtype=torch.float) < self.mask_prob
        time_emb = time_emb * (1 - pos_time_mask.unsqueeze(-1))
        
        return image_mask, slice_mask, time_mask, image_emb, slice_emb, time_emb, pos_image_mask, pos_slice_mask, pos_time_mask
    
    def _mask_sequence(self, x: torch.Tensor, bs: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_mask = torch.rand((bs * self.image_time * self.image_slice, (self.image_height // self.patch_size) * (self.image_width // self.patch_size)), dtype=torch.float, device=x.device) < self.mask_prob
        slice_mask = torch.rand((bs * self.image_time, self.image_slice), dtype=torch.float, device=x.device) < self.mask_prob
        time_mask = torch.rand((bs, self.image_time), dtype=torch.float, device=x.device) < self.mask_prob
        return image_mask, slice_mask, time_mask
    
    def forward(self, x: torch.Tensor, dimensions: torch.Tensor, seqMask: bool = False, posMask: bool = False, train: bool = False, posOutputs: bool = False):
        # Reshape and permute the input tensor
        bs = x.shape[0]
        x = self._process_input(x)
        if seqMask:
            image_mask, slice_mask, time_mask = self._mask_sequence(x, bs)
        else:
            image_mask = torch.zeros((bs * self.image_time * self.image_slice, (self.image_height // self.patch_size) * (self.image_width // self.patch_size)), dtype=torch.bool, device=x.device)
            slice_mask = torch.zeros((bs * self.image_time, self.image_slice), dtype=torch.bool, device=x.device)
            time_mask = torch.zeros((bs, self.image_time), dtype=torch.bool, device=x.device)
            
        if posMask:
            orig_image_mask, orig_slice_mask, orig_time_mask, image_emb, slice_emb, time_emb, pos_image_mask, pos_slice_mask, pos_time_mask = self._prepare_sequence_posMask(x, dimensions)
        else:
            orig_image_mask, orig_slice_mask, orig_time_mask, image_emb, slice_emb, time_emb = self._prepare_sequence(x, dimensions)

        # Encode the image level
        n = x.shape[0]
        
        x = ((image_mask.float().unsqueeze(-1) * self.masked_embed) + ((~image_mask.unsqueeze(-1)).float() * x)) + image_emb
        batch_image_class_token = self.image_class_token.expand(n, -1, -1)
        batch_image_class_mask = torch.zeros(n, 1).bool().to(x.device)
        x = torch.cat([batch_image_class_token, x], dim=1)
        encoder_image_mask = torch.cat([batch_image_class_mask, orig_image_mask], dim=1)
        image_output = self.image_encoder(x, encoder_image_mask)
        image_cls = image_output[:, 0]
        x = image_cls.view(-1, self.image_slice, self.representation_size)

        # Encode the slice level
        n = x.shape[0]
        x = ((slice_mask.float().unsqueeze(-1) * self.masked_embed) + ((~slice_mask.unsqueeze(-1)).float() * x)) + slice_emb
        batch_slice_class_token = self.slice_class_token.expand(n, -1, -1)
        batch_slice_class_mask = torch.zeros(n, 1).bool().to(x.device)
        x = torch.cat([batch_slice_class_token, x], dim=1)
        encoder_slice_mask = torch.cat([batch_slice_class_mask, orig_slice_mask], dim=1)
        slice_output = self.slice_encoder(x, encoder_slice_mask)
        slice_cls = slice_output[:, 0]
        x = slice_cls.view(-1, self.image_time, self.representation_size)
        
        # Encode the time level
        n = x.shape[0]
        x = ((time_mask.float().unsqueeze(-1) * self.masked_embed) + ((~time_mask.unsqueeze(-1)).float() * x)) + time_emb
        batch_time_class_token = self.time_class_token.expand(n, -1, -1)
        batch_time_class_mask = torch.zeros(n, 1).bool().to(x.device)
        x = torch.cat([batch_time_class_token, x], dim=1)
        encoder_time_mask = torch.cat([batch_time_class_mask, orig_time_mask], dim=1)
        time_output = self.time_encoder(x, encoder_time_mask)
        time_cls = time_output[:, 0]
        
        if train:
            cls = time_cls
            image_output = image_output[:, 1:].reshape(bs, -1, self.representation_size)
            image_mask = image_mask.reshape(bs, -1)
            slice_output = slice_output[:, 1:].reshape(bs, -1, self.representation_size)
            orig_image_mask = orig_image_mask.reshape(bs, -1)
            slice_mask = slice_mask.reshape(bs, -1)
            time_output = time_output[:, 1:].reshape(bs, -1, self.representation_size)
            orig_slice_mask = orig_slice_mask.reshape(bs, -1)
            time_mask = time_mask.reshape(bs, -1)
            output = torch.cat([image_output, slice_output, time_output], dim=1)
            orig_time_mask = orig_time_mask.reshape(bs, -1)
            mask = torch.cat([image_mask, slice_mask, time_mask], dim=1)
            orig_mask = torch.cat([orig_image_mask, orig_slice_mask, orig_time_mask], dim=1)
            if posOutputs:
                if posMask:
                    pos_image_mask = pos_image_mask.reshape(bs, -1)
                    pos_slice_mask = pos_slice_mask.reshape(bs, -1)
                    pos_time_mask = pos_time_mask.reshape(bs, -1)
                    pos_mask = torch.cat([pos_image_mask, pos_slice_mask, pos_time_mask], dim=1)
                    return self.cls_head(cls), self.embed_head(output), self.pos_head(output), mask, orig_mask, pos_mask
                
                return self.cls_head(cls), self.embed_head(output), self.pos_head(output), mask, orig_mask
            
            return self.cls_head(cls), self.embed_head(output), mask, orig_mask
        
        return x
    
    def embed(self, x: torch.Tensor, dimensions: torch.Tensor, train: bool = False, swapPos: bool = False, adversarialPos = None, adversarialPatch=None):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        
        image_mask, slice_mask, time_mask, image_emb, slice_emb, time_emb = self._prepare_sequence(x, dimensions, swapPos, adversarialPos, adversarialPatch)
        
        # Encode the image level
        n = x.shape[0]
        x = x + image_emb
        batch_image_class_token = self.image_class_token.expand(n, -1, -1)
        batch_image_class_mask = torch.zeros(n, 1).bool().to(x.device)
        x = torch.cat([batch_image_class_token, x], dim=1)
        image_mask = torch.cat([batch_image_class_mask, image_mask], dim=1)
        image_output = self.image_encoder(x, image_mask)
        image_cls = image_output[:, 0]
        x = image_cls.view(-1, self.image_slice, self.representation_size)

        # Encode the slice level
        n = x.shape[0]
        x = x + slice_emb
        batch_slice_class_token = self.slice_class_token.expand(n, -1, -1)
        batch_slice_class_mask = torch.zeros(n, 1).bool().to(x.device)
        x = torch.cat([batch_slice_class_token, x], dim=1)
        slice_mask = torch.cat([batch_slice_class_mask, slice_mask], dim=1)
        slice_output = self.slice_encoder(x, slice_mask)
        slice_cls = slice_output[:, 0]
        x = slice_cls.view(-1, self.image_time, self.representation_size)
        
        # Encode the time level
        n = x.shape[0]
        x = x + time_emb
        batch_time_class_token = self.time_class_token.expand(n, -1, -1)
        batch_time_class_mask = torch.zeros(n, 1).bool().to(x.device)
        x = torch.cat([batch_time_class_token, x], dim=1)
        time_mask = torch.cat([batch_time_class_mask, time_mask], dim=1)
        time_output = self.time_encoder(x, time_mask)
        cls = time_output[:, 0]
        
        if train:
            return self.cls_head(cls)
        
        return cls