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
        super().__init__(
            in_dim,
            [mlp_dim, in_dim],
            activation_layer=nn.GELU,
            inplace=None,
            dropout=dropout,
        )

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
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
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
        hidden_dim: int,
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
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = MaskedSequential(layers.values())
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
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
        patch_prob: float = 0.5,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        extra_cls: bool = False,
    ):
        super().__init__()
        torch._assert(
            max_height_size % patch_size == 0, "Input height indivisible by patch size!"
        )
        torch._assert(
            max_width_size % patch_size == 0, "Input width indivisible by patch size!"
        )
        self.image_height = max_height_size
        self.image_width = max_width_size
        self.image_time = max_time_size
        self.image_channels = num_channels
        self.patch_size = patch_size
        self.representation_size = representation_size
        self.projection_size = projection_size
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.mask_prob = mask_prob
        self.patch_prob = patch_prob
        self.norm_layer = norm_layer
        self.extra_cls = extra_cls
        self.conv_proj = nn.Conv3d(
            in_channels=self.image_channels,
            out_channels=representation_size,
            kernel_size=(max_time_size, patch_size, patch_size),
            stride=(max_time_size, patch_size, patch_size),
        )
        self.masked_embed = nn.Parameter(torch.zeros(1, 1, representation_size))

        seq_length = (self.image_height // patch_size) * (
            self.image_width // patch_size
        )
        self.pos_embedding = nn.Embedding(seq_length, representation_size)

        # Add class tokens
        self.class_token = nn.Parameter(torch.zeros(1, 1, representation_size))
        seq_length += 1
        if self.extra_cls:
            self.time_class_tokens = nn.Parameter(
                torch.zeros(1, self.image_time, representation_size)
            )
            seq_length += self.image_time

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

        self.cls_head = ProjectionHead(representation_size, projection_size)
        self.embed_head = ProjectionHead(representation_size, projection_size)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        p = self.patch_size
        torch._assert(
            t == self.image_time,
            f"Wrong image time dimension! Expected {self.image_time} but got {t}!",
        )
        torch._assert(
            c == self.image_channels,
            f"Wrong number of image channels! Expected {self.image_channels} but got {c}!",
        )
        torch._assert(
            h == self.image_height,
            f"Wrong image height! Expected {self.image_height} but got {h}!",
        )
        torch._assert(
            w == self.image_width,
            f"Wrong image width! Expected {self.image_width} but got {w}!",
        )
        n_h = h // p
        n_w = w // p

        # (b, t, c, h, w) -> (b, c, t, h, w) -> (b, hidden, n_h * n_w)
        x = x.permute(0, 2, 1, 4, 5)
        x = self.conv_proj(x)
        x = x.reshape(b, self.representation_size, n_h * n_w)

        # (n, representation_size, (seq_len)) -> (n, (seq_len), representation_size)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def _prepare_sequence(
        self, x: torch.Tensor, dimensions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, seq_len, _ = x.shape
        patch_height = self.image_height // self.patch_size
        patch_width = self.image_width // self.patch_size

        # Generate mask based on dimensions
        mask = torch.zeros(
            (bs, patch_height, patch_width), dtype=torch.bool, device=x.device
        )
        if self.extra_cls:
            time_cls_mask = torch.zeros(
                (bs, self.image_time), dtype=torch.bool, device=x.device
            )
        for i in range(bs):
            mask[i, math.ceil(dimensions[i, 1] / self.patch_size) :, :] = True
            mask[i, :, math.ceil(dimensions[i, 2] / self.patch_size) :] = True
            if self.extra_cls:
                time_cls_mask[i, dimensions[i, 0] :] = True
        mask = mask.reshape(bs, seq_len)

        # Add positional embeddings
        pos_indices = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(pos_indices)
        x = x + pos_emb
        if self.extra_cls:
            return x, mask, time_cls_mask
        else:
            return x, mask

    def _mask_sequence(self, x: torch.Tensor, dim: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        if torch.rand(1).item() < self.patch_prob:
            # Randomly mask out a contiguous region
            mask = torch.zeros(
                (
                    bs,
                    self.image_height // self.patch_size,
                    self.image_width // self.patch_size,
                ),
                dtype=torch.bool,
                device=x.device,
            )
            for i in range(bs):
                max_h = math.ceil(dim[i, 2] / self.patch_size)
                max_w = math.ceil(dim[i, 3] / self.patch_size)
                start_h = torch.randint(0, max_h, (1,)).item()
                start_w = torch.randint(0, max_w, (1,)).item()
                end_h = start_h + torch.randint(1, max_h - start_h + 1, (1,)).item()
                end_w = start_w + torch.randint(1, max_w - start_w + 1, (1,)).item()
                mask[i, start_h:end_h, start_w:end_w] = True
            mask = mask.reshape(bs, seq_len)
        else:
            mask = (
                torch.rand((bs, seq_len), dtype=torch.float, device=x.device)
                < self.mask_prob
            )

        return mask

    def forward(
        self,
        x: torch.Tensor,
        dimensions: torch.Tensor,
        seqMask: bool = False,
        train: bool = False,
    ):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        bs, seq_len, _ = x.shape
        if seqMask:
            mask = self._mask_sequence(x, dimensions)
        else:
            mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=x.device)

        x = (mask.float().unsqueeze(-1) * self.masked_embed) + (
            (~mask.unsqueeze(-1)).float() * x
        )

        if self.extra_cls:
            x, orig_mask, time_mask = self._prepare_sequence(x, dimensions)
        else:
            x, orig_mask = self._prepare_sequence(x, dimensions)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(bs, -1, -1)
        batch_class_mask = torch.zeros(bs, 1).bool().to(x.device)
        if self.extra_cls:
            batch_time_class_tokens = self.time_class_tokens.expand(bs, -1, -1).to(
                x.device
            )
            x = torch.cat(
                [
                    batch_class_token,
                    batch_time_class_tokens,
                    x,
                ],
                dim=1,
            )
            encoder_mask = torch.cat([batch_class_mask, time_mask, orig_mask], dim=1)
        else:
            x = torch.cat([batch_class_token, x], dim=1)
            encoder_mask = torch.cat([batch_class_mask, orig_mask], dim=1)

        x = self.encoder(x, encoder_mask)

        if train:
            cls, x = x[:, 0], x[:, 1:]
            if self.extra_cls:
                return (
                    self.cls_head(cls),
                    self.embed_head(x),
                    mask,
                    orig_mask,
                    time_mask,
                )
            else:
                return self.cls_head(cls), self.embed_head(x), mask, orig_mask

        return x

    def embed(self, x: torch.Tensor, dimensions: torch.Tensor, train: bool = False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        bs = x.shape[0]
        if self.extra_cls:
            x, mask, time_mask = self._prepare_sequence(x, dimensions)
        else:
            x, mask = self._prepare_sequence(x, dimensions)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(bs, -1, -1)
        batch_class_mask = torch.zeros(bs, 1).bool().to(x.device)
        if self.extra_cls:
            batch_time_class_tokens = self.time_class_tokens.expand(bs, -1, -1)
            x = torch.cat(
                [
                    batch_class_token,
                    batch_time_class_tokens,
                    x,
                ],
                dim=1,
            )
            mask = torch.cat([batch_class_mask, time_mask, mask], dim=1)
        else:
            x = torch.cat([batch_class_token, x], dim=1)
            mask = torch.cat([batch_class_mask, mask], dim=1)

        x = self.encoder(x, mask)
        cls = x[:, 0]

        if train:
            return self.cls_head(cls)

        return cls

    def embed_patches(self, x: torch.Tensor, dimensions: torch.Tensor, train: bool = False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        bs = x.shape[0]
        if self.extra_cls:
            x, mask, time_mask = self._prepare_sequence(x, dimensions)
        else:
            x, mask = self._prepare_sequence(x, dimensions)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(bs, -1, -1)
        batch_class_mask = torch.zeros(bs, 1).bool().to(x.device)
        if self.extra_cls:
            batch_time_class_tokens = self.time_class_tokens.expand(bs, -1, -1)
            x = torch.cat(
                [
                    batch_class_token,
                    batch_time_class_tokens,
                    x,
                ],
                dim=1,
            )
            mask = torch.cat([batch_class_mask, time_mask, mask], dim=1)
        else:
            x = torch.cat([batch_class_token, x], dim=1)
            mask = torch.cat([batch_class_mask, mask], dim=1)

        x = self.encoder(x, mask)
        x = x[:, 1:]

        if train:
            return self.embed_head(x)

        return x
