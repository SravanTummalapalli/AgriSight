from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

YOLO_MODEL_PATH = MODELS_DIR / "yolo11n.pt"
FRUIT_MODEL_PATH = MODELS_DIR / "fruit_model_leakage_free_best.pth"
QUALITY_MODEL_PATH = MODELS_DIR / "quality_model_leakage_free_best.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = "mps"


# ============================================================
# MODEL SETTINGS
# ============================================================

IMAGE_SIZE = 224

FRUIT_CLASSES = [
    "Apple",
    "Banana",
    "Guava",
    "Lime",
    "Orange",
    "Pomegranate",
]

QUALITY_CLASSES = [
    "Good",
    "Bad",
    "Mixed",
]


# ============================================================
# YOLO SETTINGS
# ============================================================

YOLO_CONFIDENCE_THRESHOLD = 0.25

SUPPORTED_FRUITS = {
    "apple",
    "banana",
    "orange",
}


# ============================================================
# CAMERA SETTINGS
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

CAMERA_BACKEND = "avfoundation"


# ============================================================
# TEMPORAL SMOOTHING
# ============================================================

SMOOTHING_FRAMES = 8


# ============================================================
# DISPLAY SETTINGS
# ============================================================

TEXT_SCALE = 0.85
TEXT_THICKNESS = 3

BOX_THICKNESS = 3


# ============================================================
# DISPLAY COLORS - BGR FORMAT
# ============================================================

COLOR_GOOD = (0, 220, 0)
COLOR_BAD = (0, 0, 255)
COLOR_MIXED = (0, 165, 255)

COLOR_OBJECT = (255, 180, 0)

COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)


# ============================================================
# UI SETTINGS
# ============================================================

WINDOW_NAME = "AgriSight - Fruit + Quality Inspection"

HEADER_HEIGHT = 80
FOOTER_HEIGHT = 90


# ============================================================
# APPLICATION SETTINGS
# ============================================================

QUIT_KEY = "q"