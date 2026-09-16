import cv2

from app.config import (
    TEXT_SCALE,
    TEXT_THICKNESS,
    BOX_THICKNESS,
    COLOR_GOOD,
    COLOR_BAD,
    COLOR_MIXED,
    COLOR_OBJECT,
    COLOR_WHITE,
    COLOR_BLACK,
    HEADER_HEIGHT,
    FOOTER_HEIGHT,
    WINDOW_NAME,
)


class Display:
    """
    Handles all visual rendering for the AgriSight application.
    """

    def __init__(self):
        self.window_name = WINDOW_NAME

    def get_quality_color(self, quality):
        """
        Return the display color for a quality class.
        """

        if quality == "Good":
            return COLOR_GOOD

        if quality == "Bad":
            return COLOR_BAD

        if quality == "Mixed":
            return COLOR_MIXED

        return COLOR_WHITE

    def draw_detection(
        self,
        frame,
        detection,
        prediction=None,
    ):
        """
        Draw one detection and its prediction on the frame.
        """

        x1, y1, x2, y2 = detection["bbox"]

        # ----------------------------------------------------
        # Non-fruit object
        # ----------------------------------------------------

        if prediction is None or not prediction.get("is_fruit", False):

            color = COLOR_OBJECT

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                BOX_THICKNESS,
            )

            label = (
                f'{detection["class_name"].capitalize()} '
                f'{detection["confidence"] * 100:.1f}%'
            )

            self._draw_label(
                frame,
                label,
                x1,
                y1,
                color,
            )

            return

        # ----------------------------------------------------
        # Fruit + quality
        # ----------------------------------------------------

        fruit = prediction["fruit"]
        quality = prediction["quality"]

        fruit_confidence = prediction["fruit_confidence"]
        quality_confidence = prediction["quality_confidence"]

        quality_color = self.get_quality_color(quality)

        # Bounding box uses quality color
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            quality_color,
            BOX_THICKNESS,
        )

        # ----------------------------------------------------
        # Two-line label
        # ----------------------------------------------------

        fruit_label = (
            f"{fruit} "
            f"{fruit_confidence * 100:.1f}%"
        )

        quality_label = (
            f"{quality} "
            f"{quality_confidence * 100:.1f}%"
        )

        self._draw_label(
            frame,
            fruit_label,
            x1,
            y1,
            quality_color,
        )

        # Put quality below fruit label
        label_y = y1 + 35

        self._draw_label(
            frame,
            quality_label,
            x1,
            label_y,
            quality_color,
        )

    def draw_header(self, frame, fps):
        """
        Draw the AgriSight header.
        """

        height, width = frame.shape[:2]

        cv2.rectangle(
            frame,
            (0, 0),
            (width, HEADER_HEIGHT),
            COLOR_BLACK,
            -1,
        )

        cv2.putText(
            frame,
            "AGRISIGHT",
            (30, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            "FRUIT + QUALITY INSPECTION",
            (30, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            COLOR_WHITE,
            1,
            cv2.LINE_AA,
        )

        fps_text = f"FPS: {fps:.1f}"

        cv2.putText(
            frame,
            fps_text,
            (width - 150, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_WHITE,
            2,
            cv2.LINE_AA,
        )

    def draw_footer(
        self,
        frame,
        objects_count,
        fruits_count,
        good_count,
        bad_count,
        mixed_count,
    ):
        """
        Draw counters at the bottom of the frame.
        """

        height, width = frame.shape[:2]

        footer_y = height - FOOTER_HEIGHT

        cv2.rectangle(
            frame,
            (0, footer_y),
            (width, height),
            COLOR_BLACK,
            -1,
        )

        counters = [
            f"Objects: {objects_count}",
            f"Fruits: {fruits_count}",
            f"GOOD: {good_count}",
            f"BAD: {bad_count}",
            f"MIXED: {mixed_count}",
        ]

        x_positions = [
            30,
            250,
            450,
            650,
            850,
        ]

        for text, x in zip(counters, x_positions):

            cv2.putText(
                frame,
                text,
                (x, footer_y + 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                COLOR_WHITE,
                2,
                cv2.LINE_AA,
            )

    def show(self, frame):
        """
        Display the final frame.
        """

        cv2.imshow(
            self.window_name,
            frame,
        )

    def _draw_label(
        self,
        frame,
        text,
        x,
        y,
        color,
    ):
        """
        Draw text with a small black background for readability.
        """

        text_size, baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            TEXT_THICKNESS,
        )

        text_width, text_height = text_size

        # Keep label inside the image
        if y - text_height - baseline < 0:
            text_y = y + text_height + baseline + 5
        else:
            text_y = y - 5

        background_top = text_y - text_height - baseline
        background_bottom = text_y + baseline

        cv2.rectangle(
            frame,
            (x, background_top),
            (x + text_width + 8, background_bottom),
            COLOR_BLACK,
            -1,
        )

        cv2.putText(
            frame,
            text,
            (x + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            TEXT_SCALE,
            color,
            TEXT_THICKNESS,
            cv2.LINE_AA,
        )