# KrishiVision Architecture

## System Overview

KrishiVision is a full-stack AI-powered crop disease detection system consisting of:

1. **MobileNetV2-based CNN** for disease classification (38 classes)
2. **Grad-CAM** for explainable AI heatmaps
3. **FastAPI backend** for REST API endpoints
4. **React + Vite frontend** for user interaction
5. **Mistral LLM integration** for multilingual treatment recommendations
6. **TFLite int8 quantization** for sub-10MB CPU deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Upload    │  │  Camera     │  │  Results    │             │
│  │  Component  │──▶│  Capture    │──▶│   Card      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                          │                       │
│                                          ▼                       │
│                              ┌─────────────────────┐            │
│                              │  Language Toggle    │            │
│                              │  (EN / HI / MR)     │            │
│                              └─────────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼ HTTPS/REST
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  /health │  │ /predict │  │ /remedy  │  │   /report    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│       │            │            │             │                │
│       ▼            ▼            ▼             ▼                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    PROCESSING PIPELINE                    │  │
│  │  1. Image Validation ──────────────────────────────────┐  │  │
│  │  2. Non-Leaf Detection (Color + Texture) ────────────┐  │  │
│  │  3. TFLite Inference (MobileNetV2) ─────────────────┐  │  │
│  │  4. Confidence Threshold Check (≥70%) ──────────────┐  │  │
│  │  5. Grad-CAM Heatmap (tf-keras-vis) ────────────────┐  │  │
│  │  6. Remedy Generation (LLM + Dictionary Fallback) ──┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                        ML MODEL                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  MobileNetV2 (ImageNet)                                  │  │
│  │  ├─ Frozen Base (Phase 1)                                │  │
│  │  └─ Unfreeze Last 30 Layers (Phase 2)                    │  │
│  │  └─ Head: GAP → Dense(256, ReLU) → Dropout(0.4) → Dense(38) │  │
│  │  Input: 224×224×3 RGB                                    │  │
│  │  Output: 38-class softmax                                │  │
│  │  TFLite int8: <10MB, <100ms CPU latency                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Pipeline
```
PlantVillage (54K images, 38 classes)
         │
         ▼
┌─────────────────────────────────────────────┐
│ Stratified Split: 70% Train / 15% Val / 15% Test │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Augmentation Pipeline                        │
│ • Rotation ±25°, Flips, Zoom 0.8-1.2        │
│ • Brightness/Contrast Jitter                 │
│ • GrabCut Background Replacement (20%)       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Phase 1: Frozen Base (lr=1e-4, 20 epochs)   │
│   - EarlyStopping(patience=3)                │
│   - ReduceLROnPlateau(factor=0.5)            │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Phase 2: Fine-tune (lr=1e-5, 30 epochs)     │
│   - Unfreeze last 30 layers                  │
│   - EarlyStopping(patience=5)                │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Evaluation                                   │
│ • PlantVillage Test Set (Lab)                │
│ • PlantDoc Field Set (Real-world)            │
│ • Ablation: Baseline / +BG / +Finetune       │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ TFLite Export (int8 Quantization)           │
│ • Representative dataset calibration        │
│ • Size verification (<10MB)                  │
│ • Latency benchmark                          │
└─────────────────────────────────────────────┘
```

### Inference Pipeline
```
User Image (Upload/Camera)
         │
         ▼
┌─────────────────────────────────────────────┐
│ Preprocessing                                │
│ • Resize to 224×224                          │
│ • Normalize (ImageNet stats)                 │
│ • uint8 quantization for TFLite              │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Rejection Module                             │
│ • Format validation                          │
│ • Non-leaf detection (HSV color + texture)   │
│ • Blur detection (Laplacian variance)        │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ TFLite Inference                             │
│ • MobileNetV2 int8                           │
│ • Top-3 predictions                          │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│ Confidence Check (≥70%)                      │
│   ├─ Accepted: Generate Grad-CAM + Remedy   │
│   ├─ Unsure: Return predictions, no remedy  │
│   └─ Rejected: Return reason                │
└─────────────────────────────────────────────┘
```

## Model Architecture

### MobileNetV2 + Custom Head
```
Input (224, 224, 3)
       │
       ▼
┌─────────────────────────────────────┐
│ MobileNetV2 Base (ImageNet weights) │
│ • 53 layers, ~2.2M params           │
│ • Depthwise separable convolutions  │
│ • Inverted residuals                │
└─────────────────────────────────────┘
       │
       ▼
GlobalAveragePooling2D
       │
       ▼
Dense(256, activation='relu')
       │
       ▼
Dropout(0.4)
       │
       ▼
Dense(38, activation='softmax')
       │
       ▼
Output: Class probabilities
```

### Training Phases

| Phase | Layers Trainable | Learning Rate | Epochs | Purpose |
|-------|-----------------|---------------|--------|---------|
| 1 | Head only (base frozen) | 1e-4 | 20 | Learn task-specific features |
| 2 | Last 30 layers + head | 1e-5 | 30 | Fine-tune for domain adaptation |

### Augmentation Strategy

| Augmentation | Probability | Parameters |
|-------------|-------------|------------|
| Horizontal Flip | 0.5 | - |
| Vertical Flip | 0.2 | - |
| Random Rotate 90° | 0.3 | - |
| Rotation | 0.7 | ±25° |
| Random Scale | 0.5 | 0.8-1.2x |
| Brightness/Contrast | 0.5 | ±20% |
| Hue/Saturation/Value | 0.3 | H±10, S±20, V±10 |
| Gauss Noise | 0.2 | var 10-50 |
| Gaussian Blur | 0.1 | kernel 3 |
| **Background Replacement** | **0.2** | **GrabCut + outdoor backgrounds** |

## API Endpoints

### GET /health
```json
{
  "status": "ok",
  "model_version": "1.0.0",
  "model_loaded": true,
  "tflite_available": true
}
```

### POST /predict
**Request:** multipart/form-data
- `file`: Image file (JPG/PNG)
- `language`: en|hi|mr (default: en)
- `include_heatmap`: boolean (default: true)
- `include_remedy`: boolean (default: true)

**Response:**
```json
{
  "verdict": "accepted|unsure|rejected",
  "crop": "Tomato",
  "disease": "Early Blight",
  "confidence": 0.87,
  "top_3": [
    {"class": "Tomato___Early_blight", "confidence": 0.87, "class_index": 29},
    {"class": "Tomato___Late_blight", "confidence": 0.08, "class_index": 30},
    {"class": "Tomato___Septoria_leaf_spot", "confidence": 0.03, "class_index": 31}
  ],
  "heatmap_b64": "base64...",
  "overlay_b64": "base64...",
  "reason": "Confidence 87% meets threshold 70%",
  "remedy": {...}
}
```

### POST /remedy
**Request:** form-data
- `class_name`: Full class name (e.g., "Tomato___Early_blight")
- `confidence`: Float
- `lang`: Query param (en|hi|mr)

**Response:** Structured remedy with organic, chemical, prevention, urgency, severity

### POST /report
Generates PDF report with diagnosis, heatmap, and remedies.

## Remedy System

### Dictionary Structure (38 classes)
```json
{
  "Tomato___Early_blight": {
    "crop": "Tomato",
    "disease": "Early Blight",
    "organic": "Mulch, stake, copper/sulfur, remove lower leaves",
    "chemical": "Chlorothalonil, mancozeb, azoxystrobin at first fruit",
    "prevention": "Rotation, mulch, resistant varieties, balanced fertility",
    "urgency": "medium",
    "severity": "Defoliation, fruit sunscald, yield loss"
  }
}
```

### LLM Integration
- **Model**: Mistral Small (mistral-small-latest)
- **Prompt**: Structured with crop, disease, confidence, severity
- **Validation**: JSON Schema validation
- **Fallback**: Dictionary entry with identical schema
- **Languages**: English, Hindi, Marathi

## Frontend Architecture

### Components
```
App
├── Header (Logo, Language Selector)
├── UploadCapture
│   ├── File Input (drag & drop)
│   └── Camera Capture (getUserMedia)
├── ResultCard
│   ├── Diagnosis Summary
│   ├── Grad-CAM Overlay
│   ├── Treatment Tabs (Organic/Chemical/Prevention)
│   ├── Top-3 Predictions Bar Chart
│   └── Action Buttons (Download PDF, New Scan)
└── Footer
```

### State Management
- React useState for local state
- No external state library (simple enough)
- API calls via axios with interceptors

### Styling
- Tailwind CSS for utility-first styling
- Custom primary color palette (green theme)
- Responsive design (mobile-first)
- Animations: fade-in, slide-up

## Deployment Considerations

### Model Serving
- TFLite int8 model: <10MB, runs on CPU
- No GPU required for inference
- FastAPI with uvicorn workers
- Can run on Raspberry Pi 4 / Jetson Nano

### Frontend Hosting
- Static build: `npm run build` → `dist/`
- Deploy to: Vercel, Netlify, GitHub Pages, or nginx
- Proxy API calls to backend

### Environment Variables
```bash
MISTRAL_API_KEY=your_key_here  # Optional, for LLM remedies
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Model Size (TFLite int8) | < 10 MB |
| CPU Inference Latency | < 100 ms |
| End-to-end API Response | < 500 ms |
| PlantVillage Test Accuracy | > 90% |
| PlantDoc Field Accuracy | > 70% |
| Macro F1 (Lab) | > 0.88 |
| Macro F1 (Field) | > 0.65 |

## Security

- No secrets in repository (`.env` gitignored)
- CORS configured for frontend origin
- Input validation on all endpoints
- File size limits (configured in nginx/proxy)
- Rate limiting recommended for production

## Monitoring

- Health endpoint for uptime checks
- Structured logging in FastAPI
- Model version tracking
- Latency metrics via middleware

## Future Improvements

1. **Model**: EfficientNet-B0, Vision Transformer
2. **Data**: Add more field datasets (PlantDoc, IP102)
3. **Features**: Disease severity staging, treatment scheduling
4. **Offline**: Full offline PWA with Service Workers
5. **Languages**: Add more Indian languages (Tamil, Telugu, etc.)
6. **Deployment**: Docker, Kubernetes, ONNX Runtime