import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "ml"
REMEDIES_DIR = BASE_DIR / "remedies"

PLANTVILLAGE_PATH = DATA_DIR / "plantvillage"
PLANT_DOC_PATH = DATA_DIR / "plantdoc"

MODEL_PATH = MODEL_DIR / "model.tflite"
MODEL_KERAS_PATH = MODEL_DIR / "model.keras"

REMEDY_DICT_PATH = REMEDIES_DIR / "remedy_dictionary.json"

CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___Powdery_mildew",
    "Cherry___healthy",
    "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

CONFIDENCE_THRESHOLD = 0.70
NON_LEAF_THRESHOLD = 0.50

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi"
}
DEFAULT_LANGUAGE = "en"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

PHASE1_EPOCHS = 20
PHASE1_LR = 1e-4
PHASE1_PATIENCE = 3

PHASE2_EPOCHS = 30
PHASE2_LR = 1e-5
PHASE2_PATIENCE = 5
UNFREEZE_LAYERS = 30

QUICK_EPOCHS = 2

AUGMENTATION_PROB = 0.8
BACKGROUND_REPLACEMENT_PROB = 0.2

ROTATION_RANGE = 25
ZOOM_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = (0.8, 1.2)
CONTRAST_RANGE = (0.8, 1.2)

TFLITE_QUANTIZATION = "int8"
TFLITE_TARGET_SIZE_MB = 10

MODEL_VERSION = "1.0.0"