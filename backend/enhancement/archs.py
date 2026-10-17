"""Arquitecturas de Real-ESRGAN en PyTorch puro.

Se reimplementa la red oficial ligera (mismos nombres de capa que los
pesos publicados por xinntao/Real-ESRGAN, licencia BSD-3-Clause) para cargar
los `.pth` directamente, sin depender de `basicsr`/`realesrgan`: esos paquetes
importan `torchvision.transforms.functional_tensor`, que ya no existe en
torchvision 0.26, y romperían el entorno.

* `SRVGGNetCompact`  -> realesr-general-x4v3 (1.2 M parámetros). Una cadena
  de convoluciones a la resolución de ENTRADA + PixelShuffle. Muy rápida: es
  la que permite mejorar el vídeo en vivo de forma fluida, incluso en una
  GTX 1650 (con la entrada reducida del perfil de baja VRAM).

El modelo grande RRDBNet (RealESRGAN_x4plus) solo se usaba para imágenes y
vídeos subidos; al quedar el sistema únicamente en vídeo en vivo se retiró.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3,
                 num_feat: int = 64, num_conv: int = 32, upscale: int = 4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1),
                nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1),
                     nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # aprende el residuo sobre un reescalado vecino-más-cercano
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")

