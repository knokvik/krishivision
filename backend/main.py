import base64
import io
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image

from config import (
    MODEL_PATH, CLASS_NAMES, CONFIDENCE_THRESHOLD,
    MODEL_VERSION, LANGUAGES, DEFAULT_LANGUAGE
)
from reject import RejectionModule, preprocess_image_from_bytes
from gradcam import create_gradcam_generator
from remedies.llm_remedy import get_remedy
from pdf_report import generate_pdf_report


class PredictionResponse(BaseModel):
    verdict: str
    crop: str
    disease: str
    confidence: float
    top_3: list
    heatmap_b64: Optional[str] = None
    overlay_b64: Optional[str] = None
    reason: Optional[str] = None
    remedy: Optional[dict] = None


class RemedyResponse(BaseModel):
    crop: str
    disease: str
    organic_treatment: str
    chemical_treatment: str
    prevention: str
    urgency: str
    severity: str
    language: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    tflite_available: bool


model = None
interpreter = None
gradcam_generator = None
rejector = RejectionModule()


def load_tflite_model():
    """Load TFLite model for inference."""
    global interpreter
    try:
        if MODEL_PATH.exists():
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
            interpreter.allocate_tensors()
            print(f"TFLite model loaded from {MODEL_PATH}")
            return True
    except Exception as e:
        print(f"Failed to load TFLite model: {e}")
    return False


def load_gradcam():
    """Load Keras model for Grad-CAM."""
    global gradcam_generator
    try:
        from config import MODEL_KERAS_PATH
        if MODEL_KERAS_PATH.exists():
            gradcam_generator = create_gradcam_generator(str(MODEL_KERAS_PATH))
            print("Grad-CAM generator loaded")
            return True
    except Exception as e:
        print(f"Failed to load Grad-CAM: {e}")
    return False


def predict_tflite(image: np.ndarray) -> np.ndarray:
    """Run inference on TFLite model."""
    if interpreter is None:
        raise RuntimeError("TFLite model not loaded")
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_data = image.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    if input_details[0]['dtype'] == np.uint8:
        input_data = (input_data * 255).astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    if output_details[0]['dtype'] == np.uint8:
        output = output.astype(np.float32) / 255.0
    
    return output[0]


def predict_keras(image: np.ndarray) -> np.ndarray:
    """Run inference on Keras model (fallback)."""
    global model
    if model is None:
        import tensorflow as tf
        from config import MODEL_KERAS_PATH
        model = tf.keras.models.load_model(MODEL_KERAS_PATH)
    
    input_data = image.astype(np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return model.predict(input_data, verbose=0)[0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting KrishiVision API...")
    load_tflite_model()
    load_gradcam()
    yield
    print("Shutting down...")


app = FastAPI(
    title="KrishiVision API",
    description="AI Crop Disease Detector",
    version=MODEL_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        model_version=MODEL_VERSION,
        model_loaded=interpreter is not None,
        tflite_available=MODEL_PATH.exists()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE),
    include_heatmap: bool = Form(True),
    include_remedy: bool = Form(True)
):
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    
    contents = await file.read()
    image = preprocess_image_from_bytes(contents)
    
    result = rejector.validate_image(image)
    if not result[0]:
        return PredictionResponse(
            verdict="rejected",
            crop="Unknown",
            disease="Unknown",
            confidence=0.0,
            top_3=[],
            reason=f"Invalid image: {result[1]}"
        )
    
    is_leaf, leaf_score, leaf_reason = rejector.is_leaf_image(image)
    if not is_leaf:
        return PredictionResponse(
            verdict="rejected",
            crop="Unknown",
            disease="Unknown",
            confidence=0.0,
            top_3=[],
            reason=f"Non-leaf image: {leaf_reason}"
        )
    
    try:
        predictions = predict_tflite(image)
    except Exception as e:
        print(f"TFLite inference failed, trying Keras: {e}")
        try:
            predictions = predict_keras(image)
        except Exception as e2:
            raise HTTPException(status_code=500, detail=f"Inference failed: {e2}")
    
    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        {
            "class": CLASS_NAMES[idx],
            "confidence": float(predictions[idx]),
            "class_index": int(idx)
        }
        for idx in top_indices
    ]
    
    top_prediction = top_predictions[0]
    confidence = top_prediction["confidence"]
    
    passes_conf, conf_reason = rejector.check_confidence(confidence)
    
    crop_disease = top_prediction["class"].split("___")
    crop = crop_disease[0].replace("_", " ")
    disease = crop_disease[1].replace("_", " ") if len(crop_disease) > 1 else "Unknown"
    
    heatmap_b64 = None
    overlay_b64 = None
    
    if include_heatmap and gradcam_generator and passes_conf:
        try:
            gradcam_result = gradcam_generator.predict_and_explain(image)
            heatmap_b64 = gradcam_result["heatmap_b64"]
            overlay_b64 = gradcam_result["overlay_b64"]
        except Exception as e:
            print(f"Grad-CAM generation failed: {e}")
    
    remedy = None
    if include_remedy and passes_conf:
        try:
            remedy = get_remedy(top_prediction["class"], confidence, language)
        except Exception as e:
            print(f"Remedy generation failed: {e}")
    
    if not passes_conf:
        return PredictionResponse(
            verdict="unsure",
            crop=crop,
            disease=disease,
            confidence=confidence,
            top_3=top_predictions,
            heatmap_b64=heatmap_b64,
            overlay_b64=overlay_b64,
            reason=conf_reason,
            remedy=remedy
        )
    
    return PredictionResponse(
        verdict="accepted",
        crop=crop,
        disease=disease,
        confidence=confidence,
        top_3=top_predictions,
        heatmap_b64=heatmap_b64,
        overlay_b64=overlay_b64,
        remedy=remedy
    )


@app.post("/remedy", response_model=RemedyResponse)
async def get_remedy_endpoint(
    class_name: str = Form(...),
    confidence: float = Form(...),
    language: str = Query(DEFAULT_LANGUAGE)
):
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    
    if class_name not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"Unknown class: {class_name}")
    
    remedy = get_remedy(class_name, confidence, language)
    return RemedyResponse(**remedy)


@app.post("/report")
async def generate_report(
    file: UploadFile = File(...),
    language: str = Form(DEFAULT_LANGUAGE)
):
    if language not in LANGUAGES:
        language = DEFAULT_LANGUAGE
    
    contents = await file.read()
    image = preprocess_image_from_bytes(contents)
    
    original_b64 = base64.b64encode(contents).decode('utf-8')
    
    is_leaf, leaf_score, leaf_reason = rejector.is_leaf_image(image)
    if not is_leaf:
        raise HTTPException(status_code=400, detail=f"Non-leaf image: {leaf_reason}")
    
    try:
        predictions = predict_tflite(image)
    except Exception:
        predictions = predict_keras(image)
    
    top_indices = np.argsort(predictions)[::-1][:3]
    top_predictions = [
        {
            "class": CLASS_NAMES[idx],
            "confidence": float(predictions[idx]),
            "class_index": int(idx)
        }
        for idx in top_indices
    ]
    
    top_prediction = top_predictions[0]
    confidence = top_prediction["confidence"]
    
    crop_disease = top_prediction["class"].split("___")
    crop = crop_disease[0].replace("_", " ")
    disease = crop_disease[1].replace("_", " ") if len(crop_disease) > 1 else "Unknown"
    
    passes_conf, _ = rejector.check_confidence(confidence)
    
    heatmap_b64 = None
    overlay_b64 = None
    if gradcam_generator and passes_conf:
        try:
            gradcam_result = gradcam_generator.predict_and_explain(image)
            heatmap_b64 = gradcam_result["heatmap_b64"]
            overlay_b64 = gradcam_result["overlay_b64"]
        except Exception as e:
            print(f"Grad-CAM failed: {e}")
    
    remedy = get_remedy(top_prediction["class"], confidence, language)
    
    prediction_dict = {
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
        "verdict": "accepted" if passes_conf else "unsure",
        "top_3": top_predictions
    }
    
    pdf_path = generate_pdf_report(
        prediction_dict, remedy,
        heatmap_b64, overlay_b64, original_b64,
        language
    )
    
    return FileResponse(
        pdf_path,
        media_type='application/pdf',
        filename=f"krishivision_report_{crop}_{disease}.pdf"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)