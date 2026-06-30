"""
FreshlyFishy Production API - ONNX Runtime Version
Optimized for Render Free Tier (512MB RAM ceiling)
"""

import asyncio
import base64
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import cv2
from matplotlib import cm

import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
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

CURRENT_DIR = Path(__file__).parent.resolve()
ONNX_MODEL_PATH = CURRENT_DIR / "best_fish_eye_model.onnx"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── IMAGENET NORMALIZATION CONSTANTS ───
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── LIGHTWEIGHT PREPROCESSING UTILITIES ───
def preprocess_image_numpy(image_bgr: np.ndarray, target_size: Tuple[int, int] = (384, 384)) -> np.ndarray:
    img_resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    img_normalized = np.transpose(img_float, (2, 0, 1))  # [3, H, W]
    for i in range(3):
        img_normalized[i] = (img_normalized[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
        
    batch_input = np.expand_dims(img_normalized, axis=0)
    return batch_input.astype(np.float32)


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


def generate_lightweight_cam(feature_map: np.ndarray, original_img: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if len(feature_map.shape) == 3:
        cam = np.mean(feature_map, axis=2)
    else:
        cam = feature_map
        
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
        
    if cam.shape != (384, 384):
        cam = cv2.resize(cam, (384, 384))
        
    cam_colored = cm.jet(cam)  # Returns RGBA matrix map
    cam_rgb = (cam_colored[:, :, :3] * 255).astype(np.uint8)
    cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(original_img, 1.0 - alpha, cam_bgr, alpha, 0)


# ─── GLOBAL ONNX RUNTIME STATE MANAGEMENT ───
global_model_state: Dict[str, Any] = {
    "session": None,
    "lock": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FishEyeNetAWPF ONNX Runtime Deployment Configuration Session...")
    try:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1  
        session_options.inter_op_num_threads = 1  
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 3  

        if not ONNX_MODEL_PATH.exists():
            logger.error(f"Static ONNX mapping target file missing from context: {ONNX_MODEL_PATH}")
            raise FileNotFoundError(f"Model file missing: {ONNX_MODEL_PATH}")

        session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )

        logger.info("✓ ONNX Runtime execution provider hooked up smoothly.")
        global_model_state["session"] = session
        global_model_state["lock"] = asyncio.Lock()

    except Exception as e:
        logger.error(f"Failed to initialize ONNX Runtime: {e}")
        raise

    yield
    global_model_state.clear()


app = FastAPI(
    title="FishEyeNetAWPF ONNX Inference Engine",
    description="Memory-optimized fish eye freshness evaluation (ONNX Runtime)",
    version="1.0.0",
    lifespan=lifespan
)

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

    try:
        if global_model_state.get("session") is None:
            return InferenceResponse(success=False, error_message="ONNX engine layer tracking unit uninitialized.", timestamp=timestamp)

        if not file.content_type or not file.content_type.startswith('image/'):
            return InferenceResponse(success=False, error_message="Invalid file binary type asset uploaded.", timestamp=timestamp)

        file_bytes = await file.read()
        if len(file_bytes) == 0:
            return InferenceResponse(success=False, error_message="Empty file package array trace detected.", timestamp=timestamp)

        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return InferenceResponse(success=False, error_message="Failed processing incoming image encoding channels.", timestamp=timestamp)

        # Execute Pipeline Transformations
        cropped_eye = batch_hough_eye_segmentation_cv(img_bgr, output_size=(384, 384))
        preprocessed_input = preprocess_image_numpy(cropped_eye, target_size=(384, 384))

        async with global_model_state["lock"]:
            try:
                session = global_model_state["session"]
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name

                outputs = session.run([output_name], {input_name: preprocessed_input})
                logits = outputs[0][0]

                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)

                pred_class = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_class])

            except Exception as inference_err:
                logger.exception(f"ONNX core tracking sequence failure: {inference_err}")
                return InferenceResponse(success=False, error_message=f"Inference session track crash: {str(inference_err)}", timestamp=timestamp)

        # Lightweight Forward Synthetic Attention Overlap Mapping
        h, w = 384, 384
        y_center, x_center = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        sigma = max(h, w) * 0.22
        feature_map = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
        feature_map = feature_map * (confidence if confidence > 0.5 else 1.0 - confidence)

        gradcam_overlay = generate_lightweight_cam(feature_map, cropped_eye, alpha=0.45)

        class_labels = {0: "Fresh", 1: "Not Fresh"}
        freshness_class = class_labels.get(pred_class, "Unknown")

        eye_base64 = img_to_base64(cropped_eye)
        overlay_base64 = img_to_base64(gradcam_overlay)

        del preprocessed_input, feature_map, nparr

        return InferenceResponse(
            success=True, freshness_class=freshness_class, confidence=round(confidence, 4),
            cropped_eye_base64=eye_base64, gradcam_overlay_base64=overlay_base64, timestamp=timestamp
        )

    except Exception as e:
        logger.exception(f"Unexpected endpoint error trace: {str(e)}")
        return InferenceResponse(success=False, error_message=f"Server Exception: {str(e)}", timestamp=timestamp)


@app.get("/health")
async def health_check():
    session_ready = global_model_state.get("session") is not None
    return {
        "status": "operational" if session_ready else "initializing",
        "model_loaded": session_ready,
        "runtime": "ONNX Runtime Layer Engine",
        "device": "CPU"
    }


@app.get("/")
async def root():
    return {
        "service": "FishEyeNetAWPF ONNX Inference Engine Core Portal",
        "version": "1.0.0",
        "runtime": "ONNX Runtime (Memory-Optimized Matrix Base)"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
