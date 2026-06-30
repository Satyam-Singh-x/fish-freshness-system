"""
FishEyeNetAWPF PyTorch → ONNX Export Script
Converts best_fish_eye_model.pth to best_fish_eye_model.onnx (CPU-optimized)

Run this ONCE locally to generate the .onnx file:
    python convert_to_onnx.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── REPRODUCE FISHEYE-NET ARCHITECTURE ───
class StableLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class DynamicSwinWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, shift: bool = False, attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.shift = shift
        self.norm = StableLayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def _window_partition(self, x: torch.Tensor, win_size: int):
        B, H, W, C = x.shape
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return windows.view(-1, win_size, win_size, C)

    def _window_reverse(self, windows: torch.Tensor, win_size: int, H: int, W: int):
        C = windows.shape[-1]
        B = int(windows.shape[0] / ((H // win_size) * (W // win_size)))
        x = windows.view(B, H // win_size, W // win_size, win_size, win_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
        return x

    def forward(self, x: torch.Tensor, win_size: int) -> torch.Tensor:
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        shift_size = win_size // 2
        if self.shift:
            x_padded = F.pad(x, (0, 0, shift_size, shift_size, shift_size, shift_size), mode='constant', value=0.0)
            x = x_padded[:, shift_size:shift_size + H, shift_size:shift_size + W, :]
        x_windows = self._window_partition(x, win_size)
        num_windows = x_windows.shape[0]
        x_windows = x_windows.view(num_windows, win_size * win_size, C)
        x_windows = self.norm(x_windows)
        qkv = self.qkv(x_windows).reshape(num_windows, win_size * win_size, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.clamp(attn, min=-32.0, max=32.0)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if torch.isnan(attn).any() or torch.isinf(attn).any():
            attn = torch.nan_to_num(attn, nan=1.0 / (win_size * win_size), posinf=0.0, neginf=0.0)
        out = (attn @ v).transpose(1, 2).reshape(num_windows, win_size * win_size, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.view(num_windows, win_size, win_size, C)
        x = self._window_reverse(out, win_size, H, W)
        if self.shift:
            x_padded = F.pad(x, (0, 0, shift_size, shift_size, shift_size, shift_size), mode='constant', value=0.0)
            x = x_padded[:, shift_size:shift_size + H, shift_size:shift_size + W, :]
        return x.permute(0, 3, 1, 2).contiguous()


class DSWModule(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.w_msa = DynamicSwinWindowAttention(dim, num_heads, shift=False)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.sw_msa = DynamicSwinWindowAttention(dim, num_heads, shift=True)
        self.drop_path = nn.Identity() if drop_path == 0.0 else nn.Dropout(drop_path)

    def forward(self, x: torch.Tensor, win_size: int) -> torch.Tensor:
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm1(x_norm).permute(0, 3, 1, 2).contiguous()
        x = x + self.drop_path(self.w_msa(x_norm, win_size))
        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm2(x_norm).permute(0, 3, 1, 2).contiguous()
        x = x + self.drop_path(self.sw_msa(x_norm, win_size))
        return x


class AWPFModule(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.down_p0_to_p1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1), nn.ReLU(inplace=True)
        )
        self.down_p1_to_p2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-5, momentum=0.1), nn.ReLU(inplace=True)
        )
        self.dsw_p2 = DSWModule(dim=dim, num_heads=8)
        self.dsw_p1 = DSWModule(dim=dim, num_heads=8)
        self.dsw_p0 = DSWModule(dim=dim, num_heads=8)

    def forward(self, p0: torch.Tensor) -> torch.Tensor:
        p1 = self.down_p0_to_p1(p0)
        p2 = self.down_p1_to_p2(p1)
        m2 = self.dsw_p2(p2, win_size=6)
        m2_upsampled = F.interpolate(m2, size=p1.shape[2:], mode='bilinear', align_corners=False)
        m1 = self.dsw_p1(p1 + m2_upsampled, win_size=8)
        m1_upsampled = F.interpolate(m1, size=p0.shape[2:], mode='bilinear', align_corners=False)
        m0 = self.dsw_p0(p0 + m1_upsampled, win_size=12)
        return m0


class StableBilinearPoolingHead(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1), nn.ReLU(inplace=True)
        )
        self.norm = nn.LayerNorm(64, eps=1e-6)
        self.fc = nn.Linear(64 * 64, embed_dim, bias=True)
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        x_norm = torch.norm(x_flat, p=2, dim=1, keepdim=True)
        x_norm = torch.clamp(x_norm, min=self.eps)
        x_normalized = x_flat / x_norm
        bilinear = torch.bmm(x_normalized, x_normalized.transpose(1, 2))
        signed_bilinear = torch.sign(bilinear) * torch.pow(torch.clamp(torch.abs(bilinear), min=self.eps), 0.5)
        signed_bilinear = signed_bilinear.view(B, C * C)
        signed_bilinear = F.normalize(signed_bilinear, p=2, dim=1)
        if torch.isnan(signed_bilinear).any():
            signed_bilinear = torch.nan_to_num(signed_bilinear, nan=0.0, posinf=0.0, neginf=0.0)
        out = self.fc(signed_bilinear)
        return self.dropout(out)


class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 1, stride: int = 1, use_skip: bool = True):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(expanded_channels, eps=1e-5, momentum=0.1), nn.SiLU(inplace=True),
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1),
        )
        self.use_skip = use_skip and (stride == 1) and (in_channels == out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_skip: out = out + x
        return out


class FishEyeNetAWPF(nn.Module):
    _IMG_EMBED_DIM = 1536

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1), nn.SiLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            FusedMBConvBlock(64, 64, expand_ratio=1, stride=1, use_skip=True),
            FusedMBConvBlock(64, 64, expand_ratio=1, stride=1, use_skip=True),
        )
        self.stage2 = nn.Sequential(
            FusedMBConvBlock(64, 128, expand_ratio=4, stride=2, use_skip=False),
            FusedMBConvBlock(128, 128, expand_ratio=4, stride=1, use_skip=True),
            FusedMBConvBlock(128, 128, expand_ratio=4, stride=1, use_skip=True),
            FusedMBConvBlock(128, 128, expand_ratio=4, stride=1, use_skip=True),
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256, eps=1e-5, momentum=0.1), nn.ReLU(inplace=True)
        )
        self.awpf = AWPFModule(dim=256)
        self.head = StableBilinearPoolingHead(in_channels=256, embed_dim=self._IMG_EMBED_DIM)
        self.classifier = nn.Linear(self._IMG_EMBED_DIM, num_classes, bias=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.stage1(x)
        x = self.stage2(x)
        p0 = self.adapter(x)
        m0 = self.awpf(p0)
        img_embed = self.head(m0)
        logits = self.classifier(img_embed)
        return logits.float()


# ─── CONVERSION EXECUTION ───
def convert_pytorch_to_onnx():
    # Environment-invariant relative path routing
    CURRENT_DIR = Path(__file__).parent.resolve()
    pt_weights_path = CURRENT_DIR / "best_fish_eye_model.pth"
    onnx_output_path = CURRENT_DIR / "best_fish_eye_model.onnx"
    input_shape = (1, 3, 384, 384)

    logger.info("=" * 60)
    logger.info("FishEyeNetAWPF Workspace Pipeline Conversion Process Initiated")
    logger.info("=" * 60)

    device = torch.device("cpu")
    model = FishEyeNetAWPF(num_classes=2)
    model = model.to(device)

    logger.info(f"Scanning target file address path: {pt_weights_path}")
    if pt_weights_path.exists():
        try:
            state_dict = torch.load(str(pt_weights_path), map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logger.info("✓ Model weight parameter dictionary mounted successfully")
        except Exception as e:
            logger.error(f"Failed loading parameters: {e}")
            return False
    else:
        logger.error("✗ Target binary .pth configuration matrix missing from folder.")
        return False

    model.eval()
    dummy_input = torch.randn(*input_shape, device=device)

    try:
        torch.onnx.export(
            model, dummy_input, str(onnx_output_path),
            input_names=["input"], output_names=["output"],
            opset_version=14, do_constant_folding=True, verbose=False,
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        logger.info(f"✓ ONNX translation matrix successfully written to: {onnx_output_path}")
        return True
    except Exception as e:
        logger.error(f"✗ Export execution crash: {e}")
        return False

if __name__ == "__main__":
    success = convert_pytorch_to_onnx()
    sys.exit(0 if success else 1)