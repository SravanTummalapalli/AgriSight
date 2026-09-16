import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from app.config import (
    QUALITY_MODEL_PATH,
    QUALITY_CLASSES,
    IMAGE_SIZE,
    DEVICE,
)


class QualityClassifier:
    """
    MobileNetV3-Large based fruit quality classifier.
    """

    def __init__(self):
        self.device = torch.device(DEVICE)

        # ----------------------------------------------------
        # Create MobileNetV3-Large architecture
        # ----------------------------------------------------

        self.model = models.mobilenet_v3_large(
            weights=None
        )

        # Replace ImageNet classifier with our 3 quality classes
        self.model.classifier[3] = nn.Linear(
            self.model.classifier[3].in_features,
            len(QUALITY_CLASSES),
        )

        # ----------------------------------------------------
        # Load trained AgriSight model
        # ----------------------------------------------------

        checkpoint = torch.load(
            QUALITY_MODEL_PATH,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        # ----------------------------------------------------
        # Image preprocessing
        # ----------------------------------------------------

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict(self, image):
        """
        Predict fruit quality.

        Args:
            image: PIL Image or OpenCV BGR image.

        Returns:
            Dictionary containing:
                quality
                confidence
                class_id
        """

        # ----------------------------------------------------
        # Convert OpenCV BGR image to PIL RGB
        # ----------------------------------------------------

        if not isinstance(image, Image.Image):
            image = Image.fromarray(
                image[:, :, ::-1]
            )

        image = image.convert("RGB")

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0).to(self.device)

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        with torch.no_grad():
            outputs = self.model(tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, class_index = torch.max(
                probabilities,
                dim=1
            )

        class_id = int(class_index.item())
        confidence = float(confidence.item())

        quality = QUALITY_CLASSES[class_id]

        return {
            "quality": quality,
            "confidence": confidence,
            "class_id": class_id,
        }