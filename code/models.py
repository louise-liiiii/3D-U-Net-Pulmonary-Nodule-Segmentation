import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be positive, got {reduction}")
        mid = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, channels, 1, bias=False),
        )

    def forward(self, x):
        avg = F.adaptive_avg_pool3d(x, 1)
        mx  = F.adaptive_max_pool3d(x, 1)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx))
        return x * attn

class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        pad = kernel_size // 2
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class CBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, sa_kernel=7):
        super().__init__()
        self.ca = ChannelAttention3D(channels, reduction=reduction)
        self.sa = SpatialAttention3D(kernel_size=sa_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


def conv3d_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),

        nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet3D_Real(nn.Module):
    """
    标准的3D U-Net（4层）
    输入:  (B,1,64,64,64)
    输出:  (B,1,64,64,64) logits
    """
    def __init__(self, in_ch=1, out_ch=1, base_ch=16, use_cbam=True):
        super().__init__()

        # encoder  输出通道

        self.enc1 = conv3d_block(in_ch, base_ch)          # 16
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = conv3d_block(base_ch, base_ch*2)      # 32
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = conv3d_block(base_ch*2, base_ch*4)    # 64
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = conv3d_block(base_ch*4, base_ch*8)    # 128
        self.pool4 = nn.MaxPool3d(2)

        # ===== CBAM on skip features =====
        self.use_cbam = use_cbam

        if self.use_cbam:
            self.cbam1 = CBAM3D(base_ch)        # 16
            self.cbam2 = CBAM3D(base_ch * 2)    # 32
            self.cbam3 = CBAM3D(base_ch * 4)    # 64
            self.cbam4 = CBAM3D(base_ch * 8)    # 128



        # bottleneck
        self.bottleneck = conv3d_block(base_ch*8, base_ch*16)  # 256

        # decoder
        self.up4 = nn.ConvTranspose3d(base_ch*16, base_ch*8, 2, 2)
        self.dec4 = conv3d_block(base_ch*16, base_ch*8)

        self.up3 = nn.ConvTranspose3d(base_ch*8, base_ch*4, 2, 2)
        self.dec3 = conv3d_block(base_ch*8, base_ch*4)

        self.up2 = nn.ConvTranspose3d(base_ch*4, base_ch*2, 2, 2)
        self.dec2 = conv3d_block(base_ch*4, base_ch*2)

        self.up1 = nn.ConvTranspose3d(base_ch*2, base_ch, 2, 2)
        self.dec1 = conv3d_block(base_ch*2, base_ch)

        self.out = nn.Conv3d(base_ch, out_ch, 1)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(f"Input must be 5D [B,C,D,H,W], got shape={tuple(x.shape)}")
        if x.shape[1] != 1:
            raise ValueError(f"Input channel must be 1, got shape={tuple(x.shape)}")

        d, h, w = x.shape[2:]
        if d % 16 != 0 or h % 16 != 0 or w % 16 != 0:
            raise ValueError(f"Input spatial size must be multiples of 16, got {(d, h, w)}")

        # encoder
        e1 = self.enc1(x)      # (B,16,64,64,64)
        p1 = self.pool1(e1)    # (B,16,32,32,32)

        e2 = self.enc2(p1)     # (B,32,32,32,32)
        p2 = self.pool2(e2)    # (B,32,16,16,16)

        e3 = self.enc3(p2)     # (B,64,16,16,16)
        p3 = self.pool3(e3)    # (B,64,8,8,8)

        e4 = self.enc4(p3)     # (B,128,8,8,8)
        p4 = self.pool4(e4)    # (B,128,4,4,4)

        b = self.bottleneck(p4)  # (B,256,4,4,4)

        # ===== CBAM refine skip features =====
        if self.use_cbam:
            e1 = self.cbam1(e1)
            e2 = self.cbam2(e2)
            e3 = self.cbam3(e3)
            e4 = self.cbam4(e4)



        # decoder
        u4 = self.up4(b)                 # (B,128,8,8,8)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)                # (B,64,16,16,16)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)                # (B,32,32,32,32)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)                # (B,16,64,64,64)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return self.out(d1)
