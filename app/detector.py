from ultralytics import YOLO

from app.config import (
    YOLO_MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
)


class ObjectDetector:
    """
    Handles YOLO object detection.
    """

    def __init__(self):
        self.model = YOLO(str(YOLO_MODEL_PATH))

    def detect(self, frame):
        """
        Run YOLO detection on a single frame.

        Args:
            frame: OpenCV BGR image.

        Returns:
            List of detected objects.
        """

        results = self.model(
            frame,
            conf=YOLO_CONFIDENCE_THRESHOLD,
            verbose=False,
        )

        result = results[0]

        detections = []

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2),
                }
            )

        return detections