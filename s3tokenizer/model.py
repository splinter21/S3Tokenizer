# Copyright (c) 2023 OpenAI. (authors: Whisper Team)
#               2024 Tsinghua Univ. (authors: Xingchen Song)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Modified from https://github.com/openai/whisper/blob/main/whisper/model.py
   Add EuclideanCodebook & VectorQuantization
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .utils import make_non_pad_mask, mask_to_bias, onnx2torch


@dataclass
class ModelDimensions:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 4096


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [
        d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)
    ]
    return freqs_cis.view(*shape)


def round_ste(z: Tensor) -> Tensor:
    """Round with straight through gradients."""
    zhat = z.round()
    return z + (zhat - z).detach()


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta**(torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


class LayerNorm(nn.LayerNorm):

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):

    def _conv_forward(self, x: Tensor, weight: Tensor,
                      bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype))


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment *
                               torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[
        np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):

    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        freqs_cis: Optional[Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask, freqs_cis)
        return self.out(wv), qk

    def qkv_attention(self,
                      q: Tensor,
                      k: Tensor,
                      v: Tensor,
                      mask: Optional[Tensor] = None,
                      freqs_cis: Optional[torch.Tensor] = None):
        _, T, D = q.shape
        scale = (D // self.n_head)**-0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        q = q.permute(0, 2, 1, 3) * scale
        k = k.permute(0, 2, 1, 3) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k  # (B, n_head, T, T)
        if mask is not None:
            qk = qk + mask
        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class ResidualAttentionBlock(nn.Module):

    def __init__(self, n_state: int, n_head: int):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(),
                                 Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask)[0]
        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoder(nn.Module):

    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        rope: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.conv1 = Conv1d(n_mels,
                            n_state,
                            kernel_size=3,
                            stride=stride,
                            padding=1)
        self.conv2 = Conv1d(n_state,
                            n_state,
                            kernel_size=3,
                            stride=2,
                            padding=1)
        self.rope = True
        if not rope:
            self.register_buffer("positional_embedding",
                                 sinusoids(n_ctx, n_state))
        else:
            freqs_cis = precompute_freqs_cis(n_mels, 1024 * 2)
            self.register_buffer("positional_embedding", freqs_cis)

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head) for _ in range(n_layer)])

    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        T = x.size(-1)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        mask = make_non_pad_mask(x_len, T).unsqueeze(1)  # (B, 1, T)
        mask = mask[:, :, (T + 1) % 2::2]  # (B, 1, T // 2)
        if self.stride == 2:
            _T = mask.size(-1)
            mask = mask[:, :, (_T + 1) % 2::2]  # (B, 1, T // 4)
        mask = mask_to_bias(mask, x.dtype)

        if not self.rope:
            x = (x + self.positional_embedding[:x.shape[1], :]).to(x.dtype)

        for block in self.blocks:
            if self.rope:
                x = block(x, mask.unsqueeze(1),
                          self.positional_embedding[:x.size(1)])
            else:
                x = block(x, mask.unsqueeze(1))

        x_len = (x_len + 1) // 2
        if self.stride == 2:
            x_len = (x_len + 1) // 2
        return x, x_len


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance (inference-only).
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
    """

    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        embed = torch.zeros(codebook_size, dim)
        self.codebook_size = codebook_size
        self.register_buffer("embed", embed)

    @torch.inference_mode()
    def preprocess(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... d -> (...) d")
        return x

    @torch.inference_mode()
    def quantize(self, x: Tensor) -> Tensor:
        embed = self.embed.t()
        dist = -(x.pow(2).sum(1, keepdim=True) - 2 * x @ embed +
                 embed.pow(2).sum(0, keepdim=True))
        embed_ind = dist.max(dim=-1).indices
        return embed_ind

    @torch.inference_mode()
    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    @torch.inference_mode()
    def dequantize(self, embed_ind: Tensor) -> Tensor:
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        embed_ind = self.quantize(x)
        # post-process
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    @torch.inference_mode()
    def decode(self, embed_ind: Tensor) -> Tensor:
        quantize = self.dequantize(embed_ind)
        return quantize


class FSQCodebook(nn.Module):

    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = torch.nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    @torch.inference_mode()
    def preprocess(self, x: Tensor) -> Tensor:
        x = rearrange(x, "... d -> (...) d")
        return x

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize

        h = x.tanh()
        h = (self.level * h).round()  # range [-k, k]
        base = 2 * self.level + 1
        powers = torch.pow(base, torch.arange(shape[-1], device=x.device))
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)
        ind = mu.reshape(shape[0], shape[1])
        return ind

    @torch.inference_mode()
    def decode(self, embed_ind: Tensor) -> Tensor:
        raise NotImplementedError(
            'There is no official up project component provided')


class VectorQuantization(nn.Module):
    """Vector quantization implementation (inference-only).
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
    """

    def __init__(self, dim: int, codebook_size: int, quantize_type='vq'):
        super().__init__()
        if quantize_type == 'vq':
            self._codebook = EuclideanCodebook(dim=dim,
                                               codebook_size=codebook_size)
        else:
            assert quantize_type == 'fsq'
            assert 3**8 == codebook_size
            self._codebook = FSQCodebook(dim=dim, level=3)
        self.quantize_type = quantize_type
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: Tensor) -> Tensor:
        if self.quantize_type == 'vq':
            x = F.normalize(x, p=2, dim=-1)
            embed_in = self._codebook.encode(x)
            return embed_in
        else:
            assert self.quantize_type == 'fsq'
            return self._codebook.encode(x)

    @torch.inference_mode()
    def decode(self, embed_ind: Tensor) -> Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize


class S3Tokenizer(nn.Module):
    """S3 tokenizer implementation (inference-only).
    Args:
        dims (ModelDimensions): Dimension
    """

    def __init__(self, name: str, dims: ModelDimensions = ModelDimensions()):
        super().__init__()
        if 'v1' not in name:
            assert 'v2' in name
            # TODO(Mddct): make it configureable
            dims.n_codebook_size = 3**8
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            2 if name
            in ["speech_tokenizer_v1_25hz", "speech_tokenizer_v2_25hz"] else 1,
        )
        self.quantizer = VectorQuantization(
            self.dims.n_audio_state,
            self.dims.n_codebook_size,
            quantize_type='vq' if 'v1' in name else 'fsq')

    def forward(self, mel: Tensor, mel_len: Tensor) -> Tuple[Tensor, Tensor]:
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(self, mel: Tensor, mel_len: Tensor) -> Tuple[Tensor, Tensor]:
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden)
        return code, code_len

    @property
    def device(self):
        return next(self.parameters()).device

    def init_from_onnx(self, onnx_path: str):
        ckpt = onnx2torch(onnx_path, None, False)
        self.load_state_dict(ckpt, strict=True)

    def init_from_pt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True)
        self.load_state_dict(ckpt, strict=True)

    def freeze(self):
        for _, param in self.named_parameters():
            param.requires_grad = False
