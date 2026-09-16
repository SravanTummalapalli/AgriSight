import cv2

from app.config import (
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CAMERA_BACKEND,
)


class Camera:
    """
    Handles camera initialization, frame capture, and cleanup.
    """

    def __init__(self):
        self.camera = None

    def open(self):
        """
        Open the camera and configure its resolution.
        """

        if CAMERA_BACKEND == "avfoundation":
            self.camera = cv2.VideoCapture(
                CAMERA_INDEX,
                cv2.CAP_AVFOUNDATION
            )
        else:
            self.camera = cv2.VideoCapture(CAMERA_INDEX)

        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )

        return self.camera

    def read(self):
        """
        Read a single frame from the camera.

        Returns:
            frame if successful, otherwise None.
        """

        if self.camera is None:
            raise RuntimeError("Camera is not open")

        success, frame = self.camera.read()

        if not success:
            return None

        return frame

    def release(self):
        """
        Release the camera.
        """

        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def is_opened(self):
        """
        Check whether the camera is currently open.
        """

        return (
            self.camera is not None
            and self.camera.isOpened()
        )