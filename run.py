import time
from collections import Counter

import cv2

from app.camera import Camera
from app.detector import ObjectDetector
from app.fruit_classifier import FruitClassifier
from app.quality_classifier import QualityClassifier
from app.predictor import Predictor
from app.smoothing import PredictionSmoother
from app.display import Display
from app.config import QUIT_KEY


def main():
    print("=" * 60)
    print("AGRISIGHT - FRUIT + QUALITY INSPECTION")
    print("=" * 60)

    camera = Camera()
    detector = ObjectDetector()
    fruit_classifier = FruitClassifier()
    quality_classifier = QualityClassifier()

    predictor = Predictor(
        fruit_classifier=fruit_classifier,
        quality_classifier=quality_classifier,
    )

    display = Display()

    fruit_smoother = PredictionSmoother()
    quality_smoother = PredictionSmoother()

    # --------------------------------------------------------
    # Open camera
    # --------------------------------------------------------

    try:
        camera.open()
    except RuntimeError as error:
        print(f"❌ {error}")
        return

    print("✅ Camera started")
    print(f"Press '{QUIT_KEY}' to quit")

    # --------------------------------------------------------
    # FPS tracking
    # --------------------------------------------------------

    previous_time = time.time()

    try:

        while True:

            # ------------------------------------------------
            # Read camera frame
            # ------------------------------------------------

            frame = camera.read()

            if frame is None:
                print("❌ Could not read frame")
                break

            # ------------------------------------------------
            # Calculate FPS
            # ------------------------------------------------

            current_time = time.time()

            elapsed_time = current_time - previous_time

            if elapsed_time > 0:
                fps = 1.0 / elapsed_time
            else:
                fps = 0.0

            previous_time = current_time

            # ------------------------------------------------
            # YOLO detection
            # ------------------------------------------------

            detections = detector.detect(frame)

            # ------------------------------------------------
            # Counters
            # ------------------------------------------------

            objects_count = len(detections)
            fruits_count = 0

            good_count = 0
            bad_count = 0
            mixed_count = 0

            # ------------------------------------------------
            # Process every detected object
            # ------------------------------------------------

            for detection in detections:

                prediction = predictor.predict(
                    frame,
                    detection,
                )

                # ------------------------------------------------
                # Non-fruit object
                # ------------------------------------------------

                if not prediction.get("is_fruit", False):

                    display.draw_detection(
                        frame,
                        detection,
                        prediction,
                    )

                    continue

                # ------------------------------------------------
                # Fruit detected
                # ------------------------------------------------

                fruits_count += 1

                # ------------------------------------------------
                # Smooth fruit prediction
                # ------------------------------------------------

                smoothed_fruit = fruit_smoother.update(
                    prediction["fruit"]
                )

                # ------------------------------------------------
                # Smooth quality prediction
                # ------------------------------------------------

                smoothed_quality = quality_smoother.update(
                    prediction["quality"]
                )

                # Replace raw predictions with smoothed values
                prediction["fruit"] = smoothed_fruit
                prediction["quality"] = smoothed_quality

                # ------------------------------------------------
                # Counters
                # ------------------------------------------------

                if smoothed_quality == "Good":
                    good_count += 1

                elif smoothed_quality == "Bad":
                    bad_count += 1

                elif smoothed_quality == "Mixed":
                    mixed_count += 1

                # ------------------------------------------------
                # Draw result
                # ------------------------------------------------

                display.draw_detection(
                    frame,
                    detection,
                    prediction,
                )

            # ------------------------------------------------
            # Draw header
            # ------------------------------------------------

            display.draw_header(
                frame,
                fps,
            )

            # ------------------------------------------------
            # Draw footer
            # ------------------------------------------------

            display.draw_footer(
                frame,
                objects_count,
                fruits_count,
                good_count,
                bad_count,
                mixed_count,
            )

            # ------------------------------------------------
            # Show frame
            # ------------------------------------------------

            display.show(frame)

            # ------------------------------------------------
            # Quit
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord(QUIT_KEY):
                break

    except KeyboardInterrupt:
        print("\nStopping AgriSight...")

    finally:
        camera.release()
        cv2.destroyAllWindows()

        print("✅ Camera released")
        print("AgriSight stopped.")


if __name__ == "__main__":
    main()