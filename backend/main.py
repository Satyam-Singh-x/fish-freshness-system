import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from matplotlib import cm

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── LOGGING CONFIGURATION ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ─── SERVER DIRECTORY WORKSPACE ARCHITECTURE ───
UPLOAD_DIR = Path("static/uploads")
PROCESSED_DIR = Path("static/processed")
WEIGHTS_PATH = r"D:\fish-freshness-system\backend\best_fish_eye_model.pth"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ─── NATIVE STABILIZED FISHEYE-NET ARCHITECTURE LAYER ENGINE ───
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
        qkv = self.qkv(x_windows).reshape(num_windows, win_size * win_size, 3, self.num_heads, self.head_dim).permute(2,
                                                                                                                      0,
                                                                                                                      3,
                                                                                                                      1,
                                                                                                                      4)
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
    def __init__(self, in_channels: int, out_channels: int, expand_ratio: int = 1, stride: int = 1,
                 use_skip: bool = True):
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


# ─── 🌟 ENVIRONMENT-INVARIANT CROSS-VERSION GRAD-CAM ENGINE ───
class ThreadSafeGradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hooks = []

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def register_hooks(self):
        h1 = self.target_layer.register_forward_hook(self._forward_hook)

        if hasattr(torch.utils.hooks, 'register_full_backward_hook'):
            h2 = torch.utils.hooks.register_full_backward_hook(self.target_layer, self._backward_hook)
        elif hasattr(nn.modules.module, 'register_full_backward_hook'):
            h2 = self.target_layer.register_full_backward_hook(self._backward_hook)
        else:
            h2 = self.target_layer.register_backward_hook(self._backward_hook)

        self.hooks = [h1, h2]

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_heatmap(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.register_hooks()
        try:
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            score = output[0, class_idx]

            self.model.zero_grad()
            score.backward(retain_graph=False)

            if self.activations is None or self.gradients is None:
                return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

            gradients = self.gradients[0]
            activations = self.activations[0]
            weights = gradients.mean(dim=(1, 2))

            cam = torch.zeros_like(activations[0])
            for i, w in enumerate(weights):
                cam += w * activations[i]

            cam = F.relu(cam)
            cam = cam.cpu().numpy()

            cam_max, cam_min = cam.max(), cam.min()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            return cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        finally:
            self.remove_hooks()


# ─── EXTRACTION PIPELINE PROCESSING UTILITIES ───
def batch_hough_eye_segmentation_cv(img_bgr: np.ndarray, output_size: Tuple[int, int] = (384, 384)) -> np.ndarray:
    h, w, _ = img_bgr.shape
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=int(h / 2),
        param1=50, param2=25,
        minRadius=int(min(w, h) * 0.2), maxRadius=int(min(w, h) * 0.6)
    )

    mask = np.zeros_like(gray)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cx, cy, r = circles[0, 0]
        padded_r = int(r * 1.05)
        cv2.circle(mask, (cx, cy), padded_r, 255, -1)
        center, radius = (cx, cy), padded_r
    else:
        center = (int(w / 2), int(h / 2))
        radius = int(min(w, h) * 0.38)
        cv2.circle(mask, center, radius, 255, -1)

    segmented_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    x_start, y_start = max(0, center[0] - radius), max(0, center[1] - radius)
    x_end, y_end = min(w, center[0] + radius), min(h, center[1] + radius)
    cropped_eye = segmented_bgr[y_start:y_end, x_start:x_end]

    if cropped_eye.size == 0 or cropped_eye.shape[0] == 0 or cropped_eye.shape[1] == 0:
        return cv2.resize(segmented_bgr, output_size, interpolation=cv2.INTER_CUBIC)

    return cv2.resize(cropped_eye, output_size, interpolation=cv2.INTER_CUBIC)


def img_to_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


def gradcam_to_overlay(cam: np.ndarray, original_img: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    cam_colored = cm.jet(cam)
    cam_bgr = cv2.cvtColor((cam_colored[:, :, :3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(original_img, 1.0 - alpha, cam_bgr, alpha, 0)


# ─── GLOBAL HARDWARE THREAD CONTEXT LOCKS ───
global_model_state: Dict[str, Any] = {
    "model": None,
    "device": None,
    "lock": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FishEyeNetAWPF model tracking blocks at deployment startup...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Active Device Core target mapping: {device}")

    model = FishEyeNetAWPF(num_classes=2)
    if os.path.exists(WEIGHTS_PATH):
        try:
            state_dict = torch.load(WEIGHTS_PATH, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded pretrained optimization weights cleanly from {WEIGHTS_PATH}")
        except Exception as e:
            logger.warning(f"Could not load weights from checkpoint storage: {e}")
    else:
        logger.warning(f"Weights parameters missing at file path target location: {WEIGHTS_PATH}")

    model = model.to(device)
    model.eval()

    global_model_state["model"] = model
    global_model_state["device"] = device
    global_model_state["lock"] = asyncio.Lock()
    yield

    global_model_state.clear()


app = FastAPI(
    title="FishEyeNetAWPF Inference Engine API",
    description="Microscopic fish eye freshness evaluation service.",
    version="1.0.0",
    lifespan=lifespan
)

# ─── 🌐 UNIVERSAL CORS BINDING OVERRIDE ENGINE ───
# Setting allow_origins=["*"] while allow_credentials=True crashes on strict modern browser engines.
# We set allow_origins=["*"] and turn off allow_credentials, which is perfect for public stateless binary inference.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceResponse(BaseModel):
    success: bool
    freshness_class: Optional[str] = None
    confidence: Optional[float] = None
    cropped_eye_base64: Optional[str] = None
    gradcam_overlay_base64: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: str = ""


@app.post("/infer", response_model=InferenceResponse)
async def infer_fish_eye(file: UploadFile = File(...)) -> InferenceResponse:
    timestamp = str(int(asyncio.get_event_loop().time()))
    upload_id = f"{timestamp}_{file.filename}"

    try:
        if global_model_state.get("model") is None:
            return InferenceResponse(success=False, error_message="Model state uninitialized at runtime context.",
                                     timestamp=timestamp)

        if not file.content_type.startswith('image/'):
            return InferenceResponse(success=False, error_message="Invalid asset metadata file formatting target.",
                                     timestamp=timestamp)

        file_bytes = await file.read()
        if len(file_bytes) == 0:
            return InferenceResponse(success=False, error_message="Empty binary payload tracking slice received.",
                                     timestamp=timestamp)

        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return InferenceResponse(success=False,
                                     error_message="Opencv pipeline image decoding target compilation failure.",
                                     timestamp=timestamp)

        # File persistent log writes
        cv2.imwrite(str(UPLOAD_DIR / f"{upload_id}_original.png"), img_bgr)

        # Extract Hough Crop Circle Array matrix
        cropped_eye = batch_hough_eye_segmentation_cv(img_bgr, output_size=(384, 384))
        cv2.imwrite(str(PROCESSED_DIR / f"{upload_id}_cropped.png"), cropped_eye)

        # Normalize target frame array
        normalize_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        cropped_rgb = cv2.cvtColor(cropped_eye, cv2.COLOR_BGR2RGB)
        tensor_input = normalize_transform(Image.fromarray(cropped_rgb)).unsqueeze(0).to(global_model_state["device"])

        # Thread-safe prediction + backward execution passing layout sequence
        async with global_model_state["lock"]:
            tensor_input.requires_grad = True
            logits = global_model_state["model"](tensor_input)
            probabilities = torch.softmax(logits, dim=1)

            pred_class = torch.argmax(probabilities[0]).item()
            confidence = probabilities[0, pred_class].item()

            cam_extractor = ThreadSafeGradCAM(global_model_state["model"], global_model_state["model"].adapter)
            cam = cam_extractor.generate_heatmap(tensor_input, pred_class)

        class_labels = {0: "Fresh", 1: "Not Fresh"}
        freshness_class = class_labels.get(pred_class, "Unknown")

        # Map color tracking grids and format outputs
        gradcam_overlay = gradcam_to_overlay(cam, cropped_eye, alpha=0.45)
        cv2.imwrite(str(PROCESSED_DIR / f"{upload_id}_gradcam.png"), gradcam_overlay)

        return InferenceResponse(
            success=True, freshness_class=freshness_class, confidence=round(confidence, 4),
            cropped_eye_base64=img_to_base64(cropped_eye), gradcam_overlay_base64=img_to_base64(gradcam_overlay),
            timestamp=timestamp
        )

    except Exception as e:
        logger.exception(f"Inference Engine Internal Pipeline Failure tracking map: {str(e)}")
        return InferenceResponse(success=False, error_message=f"API Exception Error: {str(e)}", timestamp=timestamp)


@app.get("/health")
async def health_check():
    model_ready = global_model_state.get("model") is not None
    return {
        "status": "operational" if model_ready else "initializing",
        "model_loaded": model_ready,
        "device": str(global_model_state.get("device"))
    }


@app.get("/")
async def root():
    return {
        "service": "FishEyeNetAWPF Production Inference Framework Engine",
        "endpoints": {"inference": "/infer (POST)", "health": "/health (GET)", "docs": "/docs"}
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")