from app.config import SUPPORTED_FRUITS


class Predictor:
    """
    Connects YOLO detection with the fruit and quality classifiers.
    """

    def __init__(
        self,
        fruit_classifier,
        quality_classifier,
    ):
        self.fruit_classifier = fruit_classifier
        self.quality_classifier = quality_classifier

    def predict(self, frame, detection):
        """
        Run fruit and quality prediction for one YOLO detection.

        Args:
            frame: OpenCV BGR image.
            detection: Detection dictionary returned by ObjectDetector.

        Returns:
            Prediction dictionary.
        """

        class_name = detection["class_name"].lower()
        confidence = detection["confidence"]

        # ----------------------------------------------------
        # Only process fruit classes supported by YOLO
        # ----------------------------------------------------

        if class_name not in SUPPORTED_FRUITS:
            return {
                "is_fruit": False,
                "detection": detection,
            }

        # ----------------------------------------------------
        # Extract bounding box
        # ----------------------------------------------------

        x1, y1, x2, y2 = detection["bbox"]

        height, width = frame.shape[:2]

        # Keep coordinates inside the image
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # Invalid bounding box
        if x2 <= x1 or y2 <= y1:
            return {
                "is_fruit": False,
                "detection": detection,
            }

        # ----------------------------------------------------
        # Crop fruit from frame
        # ----------------------------------------------------

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return {
                "is_fruit": False,
                "detection": detection,
            }

        # ----------------------------------------------------
        # Identify fruit
        # ----------------------------------------------------

        fruit_result = self.fruit_classifier.predict(crop)

        # ----------------------------------------------------
        # Predict quality
        # ----------------------------------------------------

        quality_result = self.quality_classifier.predict(crop)

        # ----------------------------------------------------
        # Return combined prediction
        # ----------------------------------------------------

        return {
            "is_fruit": True,

            "detection": detection,

            "fruit": fruit_result["fruit"],
            "fruit_confidence": fruit_result["confidence"],
            "fruit_class_id": fruit_result["class_id"],

            "quality": quality_result["quality"],
            "quality_confidence": quality_result["confidence"],
            "quality_class_id": quality_result["class_id"],
        }