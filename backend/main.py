"""
FreshlyFishy Production API - Multi-Output Activation Engine
Optimized for Render Free Tier (< 512MB RAM Ceiling)
Authentic Layer Activation Feature Mapping Configuration
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

# ─── RUNTIME PATH RESOLUTION ───
CURRENT_DIR = Path(__file__).parent.resolve()
ONNX_MODEL_PATH = CURRENT_DIR / "best_fish_eye_model_dual.onnx"

# ─── IMAGENET STANDARDIZATION TENSORS ───
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── DETECTIONS & SEGMENATION PIPELINE ───
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


def preprocess_image_numpy(image_bgr: np.ndarray, target_size: Tuple[int, int] = (384, 384)) -> np.ndarray:
    img_resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_CUBIC)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    
    img_normalized = np.transpose(img_float, (2, 0, 1))  # Convert layout to standard CHW
    for i in range(3):
        img_normalized[i] = (img_normalized[i] - IMAGENET_MEAN[i]) / IMAGENET_STD[i]
        
    return np.expand_dims(img_normalized, axis=0).astype(np.float32)


def generate_authentic_cam_overlay(feature_maps: np.ndarray, original_img: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Computes a true spatial activation intensity map derived dynamically 
    from the multi-channel structural configurations of the final adapter layer block.
    """
    # Channel-wise global averaging across the exposed [256, 96, 96] adapter matrix layout
    cam = np.mean(feature_maps, axis=0)  # Condenses feature volume down to [96, 96]
    
    # Mathematical Rectification Pathway (In-line ReLU behavior emulation)
    cam = np.maximum(cam, 0)
    
    # Scale intensity boundaries dynamically based on raw image conditions
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)
        
    # Interpolate from feature layer resolution back up to native [384, 384] frame size
    cam_resized = cv2.resize(cam, (384, 384), interpolation=cv2.INTER_LINEAR)
        
    cam_colored = cm.jet(cam_resized)  # Render 8-bit RGBA color configurations mapping arrays
    cam_rgb = (cam_colored[:, :, :3] * 255).astype(np.uint8)
    cam_bgr = cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR)
    
    # Alpha blend the visual heatmap directly onto the cropped corneal canvas
    return cv2.addWeighted(original_img, 1.0 - alpha, cam_bgr, alpha, 0)


def img_to_base64(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buffer).decode('utf-8')


# ─── RUNTIME CONTEXT ENGINE LIFECYCLE MANAGEMENT ───
global_model_state: Dict[str, Any] = {
    "session": None,
    "lock": None
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing FishEyeNetAWPF Dual-Output ONNX Session instance...")
    try:
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1  
        session_options.inter_op_num_threads = 1  
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.log_severity_level = 3  

        if not ONNX_MODEL_PATH.exists():
            logger.error(f"Critical execution barrier: target graph missing from path: {ONNX_MODEL_PATH}")
            raise FileNotFoundError(f"Model file missing: {ONNX_MODEL_PATH}")

        session = ort.InferenceSession(
            str(ONNX_MODEL_PATH),
            sess_options=session_options,
            providers=['CPUExecutionProvider']
        )

        logger.info("✓ Multi-output ONNX computational graph successfully instantiated into memory.")
        global_model_state["session"] = session
        global_model_state["lock"] = asyncio.Lock()

    except Exception as e:
        logger.error(f"Fatal error configuring server initialization lifespan context: {e}")
        raise

    yield
    global_model_state.clear()


app = FastAPI(
    title="FreshlyFishy Accurate Inference Engine Core",
    description="Stateless memory-invariant fish freshness processor extracting true structural activation maps.",
    version="1.2.0",
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
            return InferenceResponse(success=False, error_message="ONNX session instance completely uninitialized.", timestamp=timestamp)

        if not file.content_type or not file.content_type.startswith('image/'):
            return InferenceResponse(success=False, error_message="Invalid multipart file asset type payload uploaded.", timestamp=timestamp)

        file_bytes = await file.read()
        if len(file_bytes) == 0:
            return InferenceResponse(success=False, error_message="Zero bytes array trace found in multi-part buffer upload stream.", timestamp=timestamp)

        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return InferenceResponse(success=False, error_message="Image channel parsing error over input data vector.", timestamp=timestamp)

        # Execute Segmentation and Standardization Layers
        cropped_eye = batch_hough_eye_segmentation_cv(img_bgr, output_size=(384, 384))
        preprocessed_input = preprocess_image_numpy(cropped_eye, target_size=(384, 384))

        # Enforce execution synchronization lock to preserve low edge memory targets
        async with global_model_state["lock"]:
            try:
                session = global_model_state["session"]
                input_name = session.get_inputs()[0].name
                
                # Dynamically retrieve and target both output doors compiled into your graph
                output_names = [session.get_outputs()[0].name, session.get_outputs()[1].name]
                
                outputs = session.run(output_names, {input_name: preprocessed_input})
                logits = outputs[0][0]
                feature_maps = outputs[1][0]  # True structural tensor matrix: [256, 96, 96]

                # Compute final probability array
                exp_logits = np.exp(logits - np.max(logits))
                probabilities = exp_logits / np.sum(exp_logits)

                pred_class = int(np.argmax(probabilities))
                confidence = float(probabilities[pred_class])

            except Exception as graph_err:
                logger.error(f"ONNX computational forward propagation trace block failure: {graph_err}")
                return InferenceResponse(success=False, error_message=f"ONNX graph execution trace error: {str(graph_err)}", timestamp=timestamp)

        # Compute dynamic, image-dependent spatial activation overlay map
        gradcam_overlay = generate_authentic_cam_overlay(feature_maps, cropped_eye, alpha=0.45)

        class_labels = {0: "Fresh", 1: "Not Fresh"}
        freshness_class = class_labels.get(pred_class, "Unknown")

        eye_base64 = img_to_base64(cropped_eye)
        overlay_base64 = img_to_base64(gradcam_overlay)

        # In-line garbage memory clear to avoid free tier footprint inflation
        del preprocessed_input, feature_maps, nparr

        return InferenceResponse(
            success=True, freshness_class=freshness_class, confidence=round(confidence, 4),
            cropped_eye_base64=eye_base64, gradcam_overlay_base64=overlay_base64, timestamp=timestamp
        )

    except Exception as e:
        logger.exception(f"Pipeline processing failure tracking context: {str(e)}")
        return InferenceResponse(success=False, error_message=f"Server Exception: {str(e)}", timestamp=timestamp)


@app.get("/health")
async def health_check():
    session_ready = global_model_state.get("session") is not None
    return {
        "status": "operational" if session_ready else "initializing",
        "model_loaded": session_ready,
        "runtime": "ONNX Runtime Layer Accelerator Core",
        "device": "Shared CPU Sandboxed Architecture"
    }


@app.get("/")
async def root():
    return {
        "service": "FishEyeNetAWPF Dynamic Operational Activation API Backend",
        "version": "1.2.0",
        "runtime": "ONNX Matrix Core"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")