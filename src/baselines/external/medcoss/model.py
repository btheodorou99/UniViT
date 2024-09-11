import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
from functools import partial
from collections import OrderedDict
from torch.nn.modules.utils import _quadruple
from timm.models.vision_transformer import Block
from typing import Callable, Optional, Tuple, Union

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
        self.layers = MaskedSequential(layers.values())
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

class MedCoSS(nn.Module):
    def __init__(
        self,
        max_height_size: int,
        max_width_size: int,
        max_depth_size: int,
        max_time_size: int,
        num_channels: int,
        patch_size: int,
        depth_patch_size: int,
        time_patch_size: int,
        representation_size: int,
        num_layers: int,
        num_heads: int,
        projection_size: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        mask_prob: float = 0.15,
        decoder_dim: int = 512,
        num_decoder_layers: int = 8,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        torch._assert(max_height_size % patch_size == 0, "Input height indivisible by patch size!")
        torch._assert(max_width_size % patch_size == 0, "Input width indivisible by patch size!")
        self.image_height = max_height_size
        self.image_width = max_width_size
        self.image_depth = max_depth_size
        self.image_time = max_time_size
        self.image_channels = num_channels
        self.patch_size = patch_size
        self.depth_patch_size = depth_patch_size
        self.time_patch_size = time_patch_size
        self.representation_size = representation_size
        self.projection_size = projection_size
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.mask_prob = mask_prob
        self.norm_layer = norm_layer
        
        self.patchify_all = Conv4d(
            in_channels=self.image_channels, out_channels=representation_size, kernel_size=(time_patch_size, depth_patch_size, patch_size, patch_size), stride=(time_patch_size, depth_patch_size, patch_size, patch_size)
        )
        self.patchify_noTime = nn.Conv3d(
            in_channels=self.image_channels, out_channels=representation_size, kernel_size=(depth_patch_size, patch_size, patch_size), stride=(depth_patch_size, patch_size, patch_size)
        )
        self.patchify_no3D = nn.Conv3d(
            in_channels=self.image_channels, out_channels=representation_size, kernel_size=(time_patch_size, patch_size, patch_size), stride=(time_patch_size, patch_size, patch_size)
        )
        self.patchify_noTimeNo3D = nn.Conv2d(
            in_channels=self.image_channels, out_channels=representation_size, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size)
        )
        
        self.masked_embed = nn.Parameter(torch.zeros(1, 1, representation_size))
        self.seq_length_all = (
            (self.image_height // patch_size) * 
            (self.image_width // patch_size) * 
            (self.image_depth // depth_patch_size) * 
            (self.image_time // time_patch_size)
        )
        self.seq_length_noTime = (
            (self.image_height // patch_size) *
            (self.image_width // patch_size) *
            (self.image_depth // depth_patch_size)
        )
        self.seq_length_no3D = (
            (self.image_height // patch_size) *
            (self.image_width // patch_size) *
            (self.image_time // time_patch_size)
        )
        self.seq_length_noTimeNo3D = (
            (self.image_height // patch_size) *
            (self.image_width // patch_size)
        )
        self.pos_embedding_all = nn.Embedding(self.seq_length_all, representation_size)
        self.pos_embedding_noTime = nn.Embedding(self.seq_length_noTime, representation_size)
        self.pos_embedding_no3D = nn.Embedding(self.seq_length_no3D, representation_size)
        self.pos_embedding_noTimeNo3D = nn.Embedding(self.seq_length_noTimeNo3D, representation_size)
        seq_length = max(self.seq_length_all, self.seq_length_noTime, self.seq_length_no3D, self.seq_length_noTimeNo3D)

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, representation_size))
        seq_length += 1

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
            
        self.decoder_embed = nn.Linear(representation_size, decoder_dim)
        self.decoder_masked_embed = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed_all = nn.Parameter(torch.zeros(1, self.seq_len, decoder_dim))
        self.decoder_pos_embed_noTime = nn.Parameter(torch.zeros(1, self.seq_len, decoder_dim))
        self.decoder_pos_embed_no3D = nn.Parameter(torch.zeros(1, self.seq_len, decoder_dim))
        self.decoder_pos_embed_noTimeNo3D = nn.Parameter(torch.zeros(1, self.seq_len, decoder_dim))
        self.decoder = nn.ModuleList([
            Block(
                d_model = decoder_dim, 
                nhead = num_heads, 
                dim_feedforward = mlp_dim, 
                dropout=attention_dropout, 
                activation='gelu',
                batch_first=True
            )
            for i in range(num_decoder_layers)])
        self.decoder_pred_all = nn.Linear(decoder_dim, patch_size ** 4 * 3, bias=True)
        self.decoder_pred_noTime = nn.Linear(decoder_dim, patch_size ** 3 * 3, bias=True)
        self.decoder_pred_no3D = nn.Linear(decoder_dim, patch_size ** 3 * 3, bias=True)
        self.decoder_pred_noTimeNo3D = nn.Linear(decoder_dim, patch_size ** 2 * 3, bias=True)
        self.normalize_mim = False
        


    def _process_input(self, x: torch.Tensor, dimensions: torch.Tensor) -> torch.Tensor:
        b, t, s, c, h, w = x.shape
        p = self.patch_size
        p_d = self.depth_patch_size
        p_t = self.time_patch_size
        torch._assert(t == self.image_time, f"Wrong image time dimension! Expected {self.image_time} but got {t}!")
        torch._assert(s == self.image_depth, f"Wrong image depth dimension! Expected {self.image_depth} but got {s}!")
        torch._assert(c == self.image_channels, f"Wrong number of image channels! Expected {self.image_channels} but got {c}!")
        torch._assert(h == self.image_height, f"Wrong image height! Expected {self.image_height} but got {h}!")
        torch._assert(w == self.image_width, f"Wrong image width! Expected {self.image_width} but got {w}!")
        n_t = self.image_time // p_t
        n_d = self.image_depth // p_d
        n_h = h // p
        n_w = w // p

        # (b, t, s, c, h, w) -> (b, c, t, s, h, w) -> (b, hidden, n_h * n_w)
        x = x.permute(0, 3, 1, 2, 4, 5)
        x_out = torch.zeros((b, self.representation_size, self.seq_length - 1), device=x.device)
        mask_out = torch.zeros((b, self.seq_length - 1), dtype=torch.bool, device=x.device)
        mask_all = (dimensions[:, 0] > 1) & (dimensions[:, 1] > 1)
        mask_noTime = (dimensions[:, 0] == 1) & (dimensions[:, 1] > 1)
        mask_no3D = (dimensions[:, 0] > 1) & (dimensions[:, 1] == 1)
        mask_noTimeNo3D = (dimensions[:, 0] == 1) & (dimensions[:, 1] == 1)
        
        if mask_all.any():
            b_all = mask_all.sum()
            x_all = x[mask_all]
            patchified_all = self.patchify_all(x_all).reshape(-1, self.representation_size, n_h * n_w)
            x_out[mask_all, :, self.seq_length_all] = patchified_all
            mask_out_all = torch.zeros((b_all, p_t, p_d, n_h, n_w), dtype=torch.bool, device=x.device)
            dim_all = dimensions[mask_all]
            for i in range(b_all):
                mask_out_all[i, math.ceil(dim_all[i, 0] / self.time_patch_size) :, :, :, :] = True
                mask_out_all[i, :, math.ceil(dim_all[i, 1] / self.depth_patch_size) :, :, :] = True
                mask_out_all[i, :, :, math.ceil(dim_all[i, 2] / self.patch_size) :, :] = True
                mask_out_all[i, :, :, :, math.ceil(dim_all[i, 3] / self.patch_size) :] = True
            mask_out_all = mask_all.reshape(b_all, self.seq_length_all)
            mask_out[mask_all, :self.seq_length_all] = mask_out_all
            mask_out[mask_all, self.seq_length_all:] = True

        if mask_noTime.any():
            b_noTime = mask_noTime.sum()
            x_noTime = x[mask_noTime][:,:,0,:]
            patchified_noTime = self.patchify_noTime(x_noTime).reshape(-1, self.representation_size, n_h * n_w)
            x_out[mask_noTime, :, self.seq_length_noTime:] = patchified_noTime
            mask_out_noTime = torch.zeros((b_noTime, p_d, n_h, n_w), dtype=torch.bool, device=x.device)
            dim_noTime = dimensions[mask_noTime]
            for i in range(b_noTime):
                mask_out_noTime[i, math.ceil(dim_noTime[i, 1] / self.depth_patch_size) :, :, :] = True
                mask_out_noTime[i, :, math.ceil(dim_no3D[i, 2] / self.patch_size) :, :] = True
                mask_out_noTime[i, :, :, math.ceil(dim_no3D[i, 3] / self.patch_size) :] = True
            mask_out_noTime = mask_out_noTime.reshape(b_noTime, self.seq_length_noTime)
            mask_out[mask_noTime, :self.seq_length_noTime] = mask_out_noTime
            mask_out[mask_noTime, self.seq_length_noTime:] = True

        if mask_no3D.any():
            b_no3D = mask_no3D.sum()
            x_no3D = x[mask_no3D][:,:,:,0]
            patchified_no3D = self.patchify_no3D(x_no3D).reshape(-1, self.representation_size, n_h * n_w)
            x_out[mask_no3D, :, self.seq_length_no3D:] = patchified_no3D
            mask_out_no3D = torch.zeros((b_no3D, p_t, n_h, n_w), dtype=torch.bool, device=x.device)
            dim_no3D = dimensions[mask_no3D]
            for i in range(b_no3D):
                mask_out_no3D[i, math.ceil(dim_no3D[i, 0] / self.time_patch_size) :, :, :] = True
                mask_out_no3D[i, :, math.ceil(dim_no3D[i, 2] / self.patch_size) :, :] = True
                mask_out_no3D[i, :, :, math.ceil(dim_no3D[i, 3] / self.patch_size) :] = True
            mask_out_no3D = mask_out_no3D.reshape(b_no3D, self.seq_length_no3D)
            mask_out[mask_no3D, :self.seq_length_no3D] = mask_out_no3D
            mask_out[mask_no3D, self.seq_length_no3D:] = True

        if mask_noTimeNo3D.any():
            b_noTimeNo3D = mask_noTimeNo3D.sum()
            x_noTimeNo3D = x[mask_noTimeNo3D][:,:,0,0]
            patchified_noTimeNo3D = self.patchify_noTimeNo3D(x_noTimeNo3D).reshape(-1, self.representation_size, n_h * n_w)
            x_out[mask_noTimeNo3D, :, self.seq_length_noTimeNo3D:] = patchified_noTimeNo3D
            mask_noTimeNo3D = torch.zeros((b_noTimeNo3D, n_h, n_w), dtype=torch.bool, device=x.device)
            dim_noTimeNo3D = dimensions[mask_noTimeNo3D]
            for i in range(b_noTimeNo3D):
                mask_noTimeNo3D[i, math.ceil(dim_noTimeNo3D[i, 2] / self.patch_size) :, :] = True
                mask_noTimeNo3D[i, :, math.ceil(dim_noTimeNo3D[i, 3] / self.patch_size) :] = True
            mask_noTimeNo3D = mask_noTimeNo3D.reshape(b_noTimeNo3D, self.seq_length_noTimeNo3D)
            mask_out[mask_noTimeNo3D, :self.seq_length_noTimeNo3D] = mask_noTimeNo3D
            mask_out[mask_noTimeNo3D, self.seq_length_noTimeNo3D:] = True

        # (n, representation_size, (seq_len)) -> (n, (seq_len), representation_size)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x_out = x_out.permute(0, 2, 1)
        
        # Add positional embeddings
        pos_indices_all = torch.arange(self.seq_length_all, device=x.device)
        pos_indices_noTime = torch.arange(self.seq_length_noTime, device=x.device)
        pos_indices_no3D = torch.arange(self.seq_length_no3D, device=x.device)
        pos_indices_noTimeNo3D = torch.arange(self.seq_length_noTimeNo3D, device=x.device)
        pos_emb_all = self.pos_embedding_all(pos_indices_all)
        pos_emb_noTime = self.pos_embedding_noTime(pos_indices_noTime)
        pos_emb_no3D = self.pos_embedding_no3D(pos_indices_no3D)
        pos_emb_noTimeNo3D = self.pos_embedding_noTimeNo3D(pos_indices_noTimeNo3D)
        
        x[mask_all, :self.seq_length_all] = x[mask_all, :self.seq_length_all] + pos_emb_all
        x[mask_noTime, :self.seq_length_noTime] = x[mask_noTime, :self.seq_length_noTime] + pos_emb_noTime
        x[mask_no3D, :self.seq_length_no3D] = x[mask_no3D, :self.seq_length_no3D] + pos_emb_no3D
        x[mask_noTimeNo3D, :self.seq_length_noTimeNo3D] = x[mask_noTimeNo3D, :self.seq_length_noTimeNo3D] + pos_emb_noTimeNo3D

        return x_out, mask_out
    
    def _mask_sequence(self, x: torch.Tensor) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        num_mask = int(seq_len * self.mask_prob)
        mask = torch.rand((bs, seq_len), dtype=torch.float, device=x.device)
        mask = mask.topk(num_mask, dim=1).indices
        mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=x.device).scatter(1, mask, True)
        return mask
    
    def _basic_patchify(self, img, imgType):
        b, t, s, c, h, w = img.shape
        img = img.permute(0, 3, 1, 2, 4, 5)
        time_patches = 1
        depth_patches = 1
        height_patches = h // self.patch_size
        width_patches = w // self.patch_size
        if imgType == 1: # ALL
            img = img.reshape(b, c, time_patches, self.image_time, depth_patches, self.image_depth, height_patches, self.patch_size, width_patches, self.patch_size)
            img = torch.einsum('nctjdkhpwq->ntdhwjkpqc', img)
            img = img.reshape(b, time_patches * depth_patches * height_patches * width_patches, self.image_time * self.image_depth * self.patch_size * self.patch_size * c)
            return img
        elif imgType == 2: # NO TIME
            img = img[:,:,:,0,:]
            img = img.reshape(b, c, depth_patches, self.image_depth, height_patches, self.patch_size, width_patches, self.patch_size)
            img = torch.einsum('ncdkhpwq->ndhwkpqc', img)            
            img = img.reshape(b, depth_patches * height_patches * width_patches, self.image_depth * self.patch_size * self.patch_size * c)
            return img
        elif imgType == 3: # NO 3D
            img = img[:,:,:,:,0]
            img = img.reshape(b, c, time_patches, self.image_time, height_patches, self.patch_size, width_patches, self.patch_size)
            img = torch.einsum('nctkhpwq->nthwkpqc', img)
            img = img.reshape(b, time_patches * height_patches * width_patches, self.image_time * self.patch_size * self.patch_size * c)
            return img
        else:
            img = img[:,:,:,0,0] # NO TIME NO 3D
            img = img.reshape(b, c, height_patches, self.patch_size, width_patches, self.patch_size)
            img = torch.einsum('nchpwq->nhwpqc', img)
            img = img.reshape(b, height_patches * width_patches, self.patch_size * self.patch_size * c)
            return img

    def _mim_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.normalize_mim:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def _decoder_forward(self, x: torch.Tensor, img: torch.Tensor, dimensions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(x)
        x[:, 1:] = (mask.float().unsqueeze(-1) * self.decoder_masked_embed) + ((~mask.unsqueeze(-1)).float() * x)
        mask_all = (dimensions[:, 0] > 1) & (dimensions[:, 1] > 1)
        mask_noTime = (dimensions[:, 0] == 1) & (dimensions[:, 1] > 1)
        mask_no3D = (dimensions[:, 0] > 1) & (dimensions[:, 1] == 1)
        mask_noTimeNo3D = (dimensions[:, 0] == 1) & (dimensions[:, 1] == 1) 
        
        if mask_all.any():
            x_all = x[mask_all]
            x_all = x_all + self.decoder_pos_embed_all
            x[mask_all] = x_all

        if mask_noTime.any():
            x_noTime = x[mask_noTime]
            x_noTime = x_noTime + self.decoder_pos_embed_noTime
            x[mask_noTime] = x_noTime

        if mask_no3D.any():
            x_no3D = x[mask_no3D]
            x_no3D = x_no3D + self.decoder_pos_embed_no3D
            x[mask_no3D] = x_no3D

        if mask_noTimeNo3D.any():
            x_noTimeNo3D = x[mask_noTimeNo3D]
            x_noTimeNo3D = x_noTimeNo3D + self.decoder_pos_embed_noTimeNo3D
            x[mask_noTimeNo3D] = x_noTimeNo3D
        
        x = self.decoder(x)
        x[:, 1:]
        
        if mask_all.any():
            mim_mask_all = mask[mask_all]
            target_all = self._basic_patchify(img[mask_all], imgType=1)
            pred_all = self.decoder_pred_all(x[mask_all])
            loss_all = self._mim_loss(pred_all, target_all, mim_mask_all)
        else:
            loss_all = 0
            
        if mask_noTime.any():
            mim_mask_noTime = mask[mask_noTime]
            target_noTime = self._basic_patchify(img[mask_noTime], imgType=2)
            pred_noTime = self.decoder_pred_noTime(x[mask_noTime])
            loss_noTime = self._mim_loss(pred_noTime, target_noTime, mim_mask_noTime)
        else:
            loss_noTime = 0
            
        if mask_no3D.any():
            mim_mask_no3D = mask[mask_no3D]
            target_no3D = self._basic_patchify(img[mask_no3D], imgType=3)
            pred_no3D = self.decoder_pred_no3D(x[mask_no3D])
            loss_no3D = self._mim_loss(pred_no3D, target_no3D, mim_mask_no3D)
        else:
            loss_no3D = 0
            
        if mask_noTimeNo3D.any():
            mim_mask_noTimeNo3D = mask[mask_noTimeNo3D]
            target_noTimeNo3D = self._basic_patchify(img[mask_noTimeNo3D], imgType=4)
            pred_noTimeNo3D = self.decoder_pred_noTimeNo3D(x[mask_noTimeNo3D])
            loss_noTimeNo3D = self._mim_loss(pred_noTimeNo3D, target_noTimeNo3D, mim_mask_noTimeNo3D)
        else:
            loss_noTimeNo3D = 0
        
        loss = loss_all + loss_noTime + loss_no3D + loss_noTimeNo3D
        return loss
    
    def forward(self, img: torch.Tensor, dimensions: torch.Tensor, seqMask: bool = False, currMask: torch.Tensor = None, train: bool = False, buffer: bool = False):
        # Reshape and permute the input tensor
        x, orig_mask = self._process_input(img, dimensions)
        bs, seq_len, _ = x.shape
        if seqMask:
            if currMask is not None:
                mask = currMask
            else:
                mask = self._mask_sequence(x)
        else:
            mask = torch.zeros((bs, seq_len), dtype=torch.bool, device=x.device)

        x = (mask.float().unsqueeze(-1) * self.masked_embed) + ((~mask.unsqueeze(-1)).float() * x)

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(bs, -1, -1)
        batch_class_mask = torch.zeros(bs, 1).bool().to(x.device)
        x = torch.cat([batch_class_token, x], dim=1)
        encoder_mask = torch.cat([batch_class_mask, orig_mask], dim=1)

        x = self.encoder(x, encoder_mask)
        
        if train:
            if buffer:
                return x, mask
            
            loss = self._decoder_forward(x, img, dimensions, mask & ~orig_mask)
            return loss
        
        return x
    
    def embed(self, x: torch.Tensor, dimensions: torch.Tensor, train: bool = False):
        # Reshape and permute the input tensor
        x, mask = self._process_input(x, dimensions)
        bs = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(bs, -1, -1)
        batch_class_mask = torch.zeros(bs, 1).bool().to(x.device)
        x = torch.cat([batch_class_token, x], dim=1)
        mask = torch.cat([batch_class_mask, mask], dim=1)
        x = self.encoder(x, mask)
        cls = x[:, 0]
        
        if train:
            return self.cls_head(cls)
        
        return cls
    
    def embed_patches(self, img: torch.Tensor, dimensions: torch.Tensor):
        # Reshape and permute the input tensor
        x, mask = self._process_input(img, dimensions)
        bs, _, _ = x.shape

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(bs, -1, -1)
        batch_class_mask = torch.zeros(bs, 1).bool().to(x.device)
        x = torch.cat([batch_class_token, x], dim=1)
        mask = torch.cat([batch_class_mask, mask], dim=1)

        x = self.encoder(x, mask)
        x = x[:, 1:]
        return x
