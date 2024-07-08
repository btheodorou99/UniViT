import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..ops.misc import MLP
from functools import partial
from collections import OrderedDict
from typing import Callable, Optional, Tuple
from torch.nn.modules.utils import _quadruple

class Conv4d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:Union[int, tuple],
                 stride:Union[int, tuple] = (1, 1, 1, 1),
                 padding:Union[int, tuple] = (0, 0, 0, 0),
                 dilation:Union[int, tuple] = (1, 1, 1, 1),
                 groups:int = 1,
                 bias=False,
                 padding_mode:str ='zeros'):
        super(Conv4d, self).__init__()
        kernel_size = _quadruple(kernel_size)
        stride = _quadruple(stride)
        padding = _quadruple(padding)
        dilation = _quadruple(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        # Assertions for constructor arguments
        assert len(kernel_size) == 4, '4D kernel size expected!'
        assert len(stride) == 4, '4D Stride size expected!!'
        assert len(padding) == 4, '4D Padding size expected!!'
        assert len(dilation) == 4, '4D dilation size expected!'
        assert groups == 1, 'Groups other than 1 not yet implemented!'

        # Store constructor arguments
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.padding_mode = padding_mode

        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        # # # # # self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 3)

        # Construct weight and bias of 4D convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None
        self.reset_parameters()

        # Use a ModuleList to store layers to make the Conv4d layer trainable
        self.conv3d_layers = torch.nn.ModuleList()

        for i in range(self.kernel_size[0]):
            # Initialize a Conv3D layer
            conv3d_layer = nn.Conv3d(in_channels=self.in_channels,
                                     out_channels=self.out_channels,
                                     kernel_size=self.kernel_size[1::],
                                     padding=self.padding[1::],
                                     dilation=self.dilation[1::],
                                     stride=self.stride[1::],
                                     bias=False)
            conv3d_layer.weight = nn.Parameter(self.weight[:, :, i, :, :])

            # Store the layer
            self.conv3d_layers.append(conv3d_layer)

        del self.weight

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        # Define shortcut names for dimensions of input and kernel
        (Batch, _, l_i, d_i, h_i, w_i) = tuple(input.shape)
        (l_k, d_k, h_k, w_k) = self.kernel_size
        (l_p, d_p, h_p, w_p) = self.padding
        (l_d, d_d, h_d, w_d) = self.dilation
        (l_s, d_s, h_s, w_s) = self.stride

        # Compute the size of the output tensor based on the zero padding
        l_o = (l_i + 2 * l_p - (l_k) - (l_k-1) * (l_d-1))//l_s + 1
        d_o = (d_i + 2 * d_p - (d_k) - (d_k-1) * (d_d-1))//d_s + 1
        h_o = (h_i + 2 * h_p - (h_k) - (h_k-1) * (h_d-1))//h_s + 1
        w_o = (w_i + 2 * w_p - (w_k) - (w_k-1) * (w_d-1))//w_s + 1

        # Pre-define output tensors
        out = torch.zeros(Batch, self.out_channels, l_o, d_o, h_o, w_o).to(input.device)

        # Convolve each kernel frame i with each input frame j
        for i in range(l_k):
            # Calculate the zero-offset of kernel frame i
            zero_offset = - l_p + (i * l_d)
            # Calculate the range of input frame j corresponding to kernel frame i
            j_start = max(zero_offset % l_s, zero_offset)
            j_end = min(l_i, l_i + l_p - (l_k-i-1)*l_d)
            # Convolve each kernel frame i with corresponding input frame j
            for j in range(j_start, j_end, l_s):
                # Calculate the output frame
                out_frame = (j - zero_offset) // l_s
                # Add results to this output frame
                out[:, :, out_frame, :, :, :] += self.conv3d_layers[i](input[:, :, j, :, :])

        # Add bias to output
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1, 1)

        return out
    
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
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
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
        self.layers = MaskedSequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, mask: Optional[torch.Tensor] = None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
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
        self.conv_proj = Conv4d(
            in_channels=self.image_channels, out_channels=representation_size, kernel_size=(max_time_size, max_slice_size, patch_size, patch_size), stride=(max_time_size, max_slice_size, patch_size, patch_size)
        )
        self.masked_embed = nn.Parameter(torch.zeros(1, 1, representation_size))
        
        seq_length = (self.image_height // patch_size) * (self.image_width // patch_size)
        self.pos_embedding = nn.Embedding(seq_length, representation_size)

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

        # (b, t, s, c, h, w) -> (b, c, t, s, h, w) -> (b, hidden, n_h * n_w)
        x = x.permute(0, 3, 1, 2, 4, 5)
        x = self.conv_proj(x)
        x = x.reshape(b, self.representation_size, n_h * n_w)

        # (n, representation_size, (seq_len)) -> (n, (seq_len), representation_size)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def _prepare_sequence(self, x: torch.Tensor, dimensions: torch.Tensor, swapPos: bool = False, adversarialPos = None, adversarialPatch = None) -> Tuple[torch.Tensor, torch.Tensor]:
        bs, seq_len, _ = x.shape
        patch_height = self.image_height // self.patch_size
        patch_width = self.image_width // self.patch_size
        
        # Generate mask based on dimensions
        mask = torch.zeros((bs, patch_height, patch_width), dtype=torch.bool, device=x.device)
        time_cls_mask = torch.zeros((bs, self.image_time), dtype=torch.bool)
        slice_cls_mask = torch.zeros((bs, self.image_slice), dtype=torch.bool)
        for i in range(bs):
            mask[i, math.ceil(dimensions[i,2] / self.patch_size):, :] = True
            mask[i, :, math.ceil(dimensions[i,3] / self.patch_size):] = True
            time_cls_mask[i, dimensions[i,0]:] = True
            slice_cls_mask[i, dimensions[i,1]:] = True
        mask = mask.reshape(bs, seq_len)
        
        # Add positional embeddings
        pos_indices = torch.arange(self.seq_length - 1, device=x.device)
        pos_emb = self.pos_embedding(pos_indices)
        
        if swapPos:
            num_swaps = int(seq_len * self.swap_prob)
            # Generate random indices for swapping
            for b in range(bs):
                # Ensure we get unique indices for swapping
                indices_a = torch.randperm(seq_len)[:num_swaps]
                indices_b = torch.randperm(seq_len)[:num_swaps]
                temp = pos_emb[b, indices_a].clone()
                pos_emb[b, indices_a], pos_emb[b, indices_b] = pos_emb[b, indices_b], temp
        elif adversarialPos is not None:
            pos_mask = torch.rand_like(bs, seq_len, 1, dtype=torch.float) < self.mask_prob
            pos_emb[pos_mask] = adversarialPos(pos_emb[pos_mask])
        
        x = x + pos_emb
        if adversarialPatch is not None:
            patch_mask = torch.rand_like(bs, seq_len, 1, dtype=torch.float) < self.mask_prob
            x[patch_mask] = adversarialPatch(x[patch_mask])
            
        return x, mask, time_cls_mask, slice_cls_mask
    
    def _mask_sequence(self, x: torch.Tensor, dim: torch.Tensor, patchMask: bool = False) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        if patchMask:
            # Randomly mask out a contiguous region
            mask = torch.zeros((bs, self.image_height // self.patch_size, self.image_width // self.patch_size), dtype=torch.bool, device=x.device)
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
            mask = torch.rand((bs, seq_len), dtype=torch.float, device=x.device) < self.mask_prob
        
        return mask
    
    def forward(self, x: torch.Tensor, dimensions: torch.Tensor, seqMask: bool = False, train: bool = False, extraCls: bool = False, swapPos: bool = False, adversarialPos = None, adversarialPatch=None, patchMask: bool = False):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        bs, seq_len, _ = x.shape
        if seqMask:
            mask = self._mask_sequence(x, dimensions, patchMask)
        else:
            mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=x.device)
            
        x = (mask.float().unsqueeze(-1) * self.masked_embed) + ((~mask.unsqueeze(-1)).float() * x)
        x, orig_mask, time_mask, slice_mask = self._prepare_sequence(x, dimensions, swapPos, adversarialPos, adversarialPatch)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(bs, -1, -1)
        batch_class_mask = torch.zeros(bs, 1).bool().to(x.device)
        batch_time_class_tokens = self.time_class_tokens.expand(bs, -1, -1)
        batch_slice_class_tokens = self.slice_class_tokens.expand(bs, -1, -1)
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
        bs = x.shape[0]
        
        x, mask, time_mask, slice_mask = self._prepare_sequence(x, dimensions)
        
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(bs, -1, -1)
        batch_class_mask = torch.zeros(bs, 1).bool().to(x.device)
        batch_time_class_tokens = self.time_class_tokens.expand(bs, -1, -1)
        batch_slice_class_tokens = self.slice_class_tokens.expand(bs, -1, -1)
        x = torch.cat([batch_class_token, batch_time_class_tokens, batch_slice_class_tokens, x], dim=1)
        mask = torch.cat([batch_class_mask, time_mask, slice_mask, mask], dim=1)

        x = self.encoder(x, mask)
        cls = x[:, 0]
        
        if train:
            return self.cls_head(cls)
        
        return cls