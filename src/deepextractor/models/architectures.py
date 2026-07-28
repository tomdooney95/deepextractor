import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TD building blocks (used by UNET1D_LSTM_ATT)
# ---------------------------------------------------------------------------

class SE1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1D features (B, C, T)."""
    def __init__(self, channels, r=8):
        super().__init__()
        hidden = max(1, channels // r)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.net(x)


class MHSA1D(nn.Module):
    """Temporal multi-head self-attention on (B, C, T)."""
    def __init__(self, channels, num_heads=4, dropout=0.0):
        super().__init__()
        self.proj_in  = nn.Conv1d(channels, channels, 1)
        self.attn     = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.proj_out = nn.Conv1d(channels, channels, 1)
        self.norm     = nn.LayerNorm(channels)

    def forward(self, x):
        z = self.proj_in(x).transpose(1, 2)      # (B, T, C)
        z2, _ = self.attn(z, z, z)
        z = self.norm(z + z2).transpose(1, 2)    # (B, C, T)
        return self.proj_out(z)


class LSTMBottleneck1D(nn.Module):
    """Bidirectional LSTM bottleneck for (B, C, T) features."""
    def __init__(
        self,
        channels,
        hidden: Optional[int] = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden = hidden or channels // 2
        self.lstm = nn.LSTM(
            input_size=channels, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_ch = hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_ch, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        z = x.transpose(1, 2)              # (B, T, C)
        z, _ = self.lstm(z)
        z = self.norm(self.proj(z))
        return z.transpose(1, 2)           # (B, C, T)


class _DoubleConv1D_norm(nn.Module):
    """DoubleConv1D with configurable normalisation (used by UNET1D_LSTM_ATT)."""
    def __init__(self, in_ch, out_ch, norm='bn'):
        super().__init__()
        def _norm(ch):
            return nn.BatchNorm1d(ch) if norm == 'bn' else nn.GroupNorm(ch, ch)
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch), nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            _norm(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET1D_LSTM_ATT(nn.Module):
    """
    Time-domain U-Net for two-detector signal/glitch separation.

    Input : (B, in_channels, T)  — default 2 (H1 + L1 strain stacked)
    Output: (B, out_channels, T) — default 4 (h1_signal, h1_noise, l1_signal, l1_noise)

    Optional bottleneck: bidirectional LSTM, multi-head self-attention.
    Optional skip connections: Squeeze-and-Excitation (SE) gating.
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 4,
        features=(64, 128, 256, 512),
        use_lstm_bottleneck: bool = True,
        lstm_layers: int = 1,
        lstm_bidirectional: bool = True,
        lstm_dropout: float = 0.0,
        use_mhsa_bottleneck: bool = False,
        use_mhsa_encoder: bool = False,
        use_mhsa_decoder: bool = False,
        attn_heads: int = 4,
        use_se_on_skips: bool = True,
        norm: str = 'bn',
    ):
        super().__init__()
        self.pool = nn.MaxPool1d(2, 2)
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.skip_se = nn.ModuleList() if use_se_on_skips else None
        self.use_lstm = use_lstm_bottleneck
        self.use_mhsa_bottleneck = use_mhsa_bottleneck
        self.use_mhsa_encoder = use_mhsa_encoder
        self.use_mhsa_decoder = use_mhsa_decoder

        # Encoder
        ch_in = in_channels
        for f in features:
            self.downs.append(_DoubleConv1D_norm(ch_in, f, norm=norm))
            if use_se_on_skips:
                self.skip_se.append(SE1D(f))
            ch_in = f

        # Bottleneck
        bottleneck_ch = features[-1]
        self.bottleneck_conv = _DoubleConv1D_norm(bottleneck_ch, bottleneck_ch * 2, norm=norm)
        ch_bn = bottleneck_ch * 2

        if use_lstm_bottleneck:
            self.bottleneck_lstm = LSTMBottleneck1D(
                channels=ch_bn, num_layers=lstm_layers,
                bidirectional=lstm_bidirectional, dropout=lstm_dropout,
            )
        if use_mhsa_bottleneck:
            self.bottleneck_attn = MHSA1D(channels=ch_bn, num_heads=attn_heads)
        if use_mhsa_encoder:
            self.encoder_attn = MHSA1D(channels=bottleneck_ch, num_heads=attn_heads)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose1d(ch_bn, f, kernel_size=2, stride=2))
            self.ups.append(_DoubleConv1D_norm(f * 2, f, norm=norm))
            ch_bn = f

        if use_mhsa_decoder:
            self.decoder_attn = MHSA1D(channels=features[-1], num_heads=attn_heads)

        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        z = x
        for i, down in enumerate(self.downs):
            z = down(z)
            skips.append(self.skip_se[i](z) if self.skip_se is not None else z)
            z = self.pool(z)

        if self.use_mhsa_encoder:
            z = z + self.encoder_attn(z)

        z = self.bottleneck_conv(z)
        if self.use_lstm:
            z = z + self.bottleneck_lstm(z)
        if self.use_mhsa_bottleneck:
            z = z + self.bottleneck_attn(z)

        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            z = self.ups[i](z)
            skip = skips[i // 2]
            if z.shape[-1] != skip.shape[-1]:
                z = F.interpolate(z, size=skip.shape[-1])
            z = torch.cat([skip, z], dim=1)
            z = self.ups[i + 1](z)
            if self.use_mhsa_decoder and i == 0:
                z = z + self.decoder_attn(z)

        return self.final_conv(z)


# 1D models


class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(DoubleConv1D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Dropout applied after the conv block so Sequential indices are unchanged
        # — existing checkpoints load cleanly (Dropout1d has no weights/buffers).
        # Same MC-Dropout pattern as DoubleConv2D below.
        self.dropout = nn.Dropout1d(dropout_p) if dropout_p > 0.0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNET1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
                 dropout_p=0.0):
        super(UNET1D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Encoder — no dropout; learns stable feature detectors.
        for feature in features:
            self.downs.append(DoubleConv1D(in_channels, feature, dropout_p=0.0))
            in_channels = feature

        # Decoder — dropout applied here and at the bottleneck so uncertainty
        # propagates through the full reconstruction path.
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv1D(feature * 2, feature, dropout_p=dropout_p))

        self.bottleneck = DoubleConv1D(features[-1], features[-1] * 2, dropout_p=dropout_p)
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class DnCNN1D(nn.Module):
    def __init__(self, depth=12, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN1D, self).__init__()
        padding = 1
        layers = []

        layers.append(
            nn.Conv1d(
                in_channels=image_channels,
                out_channels=n_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(
                nn.Conv1d(
                    in_channels=n_channels,
                    out_channels=n_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm1d(n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=image_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        return self.dncnn(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.orthogonal_(m.weight)
                logger.debug("init weight")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class Autoencoder1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(Autoencoder1D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv1D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv1D(feature, feature))

        self.bottleneck = DoubleConv1D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        for down in self.downs:
            x = down(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# 2D models


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(DoubleConv2D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Dropout applied after the conv block so Sequential indices are unchanged
        # — existing checkpoints load cleanly (Dropout2d has no weights/buffers).
        self.dropout = nn.Dropout2d(dropout_p) if dropout_p > 0.0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNET2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512],
                 dropout_p=0.0):
        super(UNET2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder — no dropout; learns stable feature detectors.
        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature, dropout_p=0.0))
            in_channels = feature

        # Decoder — dropout applied here and at the bottleneck so uncertainty
        # propagates through the full reconstruction path.
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv2D(feature * 2, feature, dropout_p=dropout_p))

        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2, dropout_p=dropout_p)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.final_conv(x)


class Autoencoder2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512]):
        super(Autoencoder2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv2D(feature, feature))

        self.bottleneck = DoubleConv2D(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        original_size = x.size()[-2:]
        downs_outputs = []
        for down in self.downs:
            x = down(x)
            downs_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        downs_outputs = downs_outputs[::-1]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            target_size = downs_outputs[idx // 2].size()[-2:]
            if x.size()[-2:] != target_size:
                x = F.interpolate(x, size=target_size)
            x = self.ups[idx + 1](x)

        if x.size()[-2:] != original_size:
            x = F.interpolate(x, size=original_size)

        return self.final_conv(x)


class ModifiedAutoencoder2D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256]):
        super(ModifiedAutoencoder2D, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv2D(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(feature * 2, feature, kernel_size=3, stride=1, padding=1),
                )
            )

        self.bottleneck = nn.Conv2d(
            features[-1], features[-1] * 2, kernel_size=3, stride=1, padding=1
        )
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        original_size = x.size()[-2:]
        downs_outputs = []
        for down in self.downs:
            x = down(x)
            downs_outputs.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        downs_outputs = downs_outputs[::-1]
        for idx, up in enumerate(self.ups):
            x = up(x)
            target_size = downs_outputs[idx].size()[-2:]
            if x.size()[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        if x.size()[-2:] != original_size:
            x = F.interpolate(x, size=original_size)

        return self.final_conv(x)
