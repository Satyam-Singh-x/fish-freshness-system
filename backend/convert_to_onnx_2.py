import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. DYNAMIC SWIN WINDOW (DSW) ATTENTION MODULE COMPONENTS
# =========================================================================
class DynamicSwinWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, shift: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.shift = shift
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

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
            x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        x_windows = self._window_partition(x, win_size)
        num_windows = x_windows.shape[0]
        x_windows = x_windows.view(num_windows, win_size * win_size, C)
        qkv = self.qkv(x_windows).reshape(num_windows, win_size * win_size, 3, self.num_heads,
                                          C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(num_windows, win_size * win_size, C)
        out = self.proj(out)
        out = out.view(num_windows, win_size, win_size, C)
        x = self._window_reverse(out, win_size, H, W)
        if self.shift:
            x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))
        return x.permute(0, 3, 1, 2).contiguous()


class DSWModule(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.w_msa = DynamicSwinWindowAttention(dim, num_heads, shift=False)
        self.norm2 = nn.BatchNorm2d(dim)
        self.sw_msa = DynamicSwinWindowAttention(dim, num_heads, shift=True)

    def forward(self, x: torch.Tensor, win_size: int) -> torch.Tensor:
        x = x + self.w_msa(self.norm1(x), win_size)
        x = x + self.sw_msa(self.norm2(x), win_size)
        return x

# =========================================================================
# 2. ADAPTIVE WINDOW PYRAMIL FUSION (AWPF) MODULE


# =========================================================================
class AWPFModule(nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.down_p0_to_p1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True)
        )
        self.down_p1_to_p2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True)
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

# =========================================================================
# 3. SECOND-ORDER COMPACT BILINEAR POOLING EXTRACTOR HEAD


# =========================================================================
class BilinearPoolingHead(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True)
        )
        self.fc = nn.Linear(64 * 64, embed_dim)
        self.dropout = nn.Dropout(p=0.35)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compress(x)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)
        bilinear = torch.bmm(x, x.transpose(1, 2)) / (H * W)
        bilinear = bilinear.view(B, C * C)
        bilinear = torch.sign(bilinear) * torch.sqrt(torch.abs(bilinear) + 1e-8)
        bilinear = F.normalize(bilinear, p=2, dim=1)
        out = self.fc(bilinear)
        return self.dropout(out)

# =========================================================================
# 4. BACKBONE BLOCK & HYBRID BASE NET


# =========================================================================
class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 1, stride: int = 1,
                 use_skip: bool = True):
        super().__init__()
        expanded_channels = in_channels * expand_ratio
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(expanded_channels), nn.SiLU(inplace=True),
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
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
            nn.BatchNorm2d(32), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(inplace=True),
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
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.awpf = AWPFModule(dim=256)
        self.head = BilinearPoolingHead(in_channels=256, embed_dim=self._IMG_EMBED_DIM)
        self.classifier = nn.Linear(self._IMG_EMBED_DIM, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)
        x = self.stage1(x)
        x = self.stage2(x)
        p0 = self.adapter(x)
        m0 = self.awpf(p0)
        img_embed = self.head(m0)
        return self.classifier(img_embed)


# =========================================================================
# 5. DUAL-OUTPUT GRAPH EXPORT WRAPPER CLASS
# =========================================================================
class FishEyeNetAWPFONNXWrapper(nn.Module):
    """
    Exposes both the classification logits and the adapter features
    as discrete network outputs inside the unified ONNX binary layout.
    """

    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.model.eval()

    def forward(self, x):
        # Forward pass execution path matching the standard topology
        x_stem = self.model.stem(x)
        x_s1 = self.model.stage1(x_stem)
        x_s2 = self.model.stage2(x_s1)

        # Capture the raw high-resolution intermediate activation layer state
        adapter_features = self.model.adapter(x_s2)  # Shape: [Batch, 256, 96, 96]

        # Resume operations to compute logits
        m0 = self.model.awpf(adapter_features)
        img_embed = self.model.head(m0)
        logits = self.model.classifier(img_embed)  # Shape: [Batch, 2]

        # Return both tensor pointers to anchor them as explicit graph exit paths
        return logits, adapter_features


# =========================================================================
# 6. LOCAL CPU CONERT FUNCTION
# =========================================================================
def execute_local_cpu_export():
    pth_weight_path = "best_fish_eye_model.pth"
    output_onnx_path = "best_fish_eye_model_dual.onnx"

    print("Building local baseline architectural network block...")
    base_model = FishEyeNetAWPF(num_classes=2)

    print(f"Loading local weight dictionary from: {pth_weight_path}...")
    if not os.path.exists(pth_weight_path):
        raise FileNotFoundError(f"Missing weight target: '{pth_weight_path}' in the current working directory.")

    # Safely assign weights directly onto the CPU data map layout
    state_dict = torch.load(pth_weight_path, map_location=torch.device('cpu'))
    base_model.load_state_dict(state_dict)

    # Initialize the dual-output pipeline wrapper context
    onnx_wrapper = FishEyeNetAWPFONNXWrapper(base_model)

    # Create an artificial tensor input tracking baseline CHW properties
    # Shape profile: [1 Image, 3 Channels (RGB), 384px Height, 384px Width]
    dummy_input = torch.randn(1, 3, 384, 384, dtype=torch.float32)

    input_names = ["images"]
    output_names = ["logits", "adapter_features"]

    # Permit dynamic sizing constraints along the batch axis index
    dynamic_axes = {
        "images": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "adapter_features": {0: "batch_size"}
    }

    print(f"Compiling computational tracing nodes down to ONNX binary at: {output_onnx_path}...")

    torch.onnx.export(
        onnx_wrapper,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    print(f"✓ Transformation complete! Multi-output ONNX file successfully generated on local CPU.")


if __name__ == "__main__":
    execute_local_cpu_export()