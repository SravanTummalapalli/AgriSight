from collections import deque, Counter

from app.config import SMOOTHING_FRAMES


class PredictionSmoother:
    """
    Maintains separate prediction histories for different objects.
    """

    def __init__(self, max_frames=SMOOTHING_FRAMES):
        self.max_frames = max_frames
        self.histories = {}

    def update(self, object_id, prediction):
        """
        Add a prediction to the history of a specific object.

        Args:
            object_id: Identifier for the detected object.
            prediction: Prediction label.

        Returns:
            Smoothed prediction.
        """

        if prediction is None:
            return None

        if object_id not in self.histories:
            self.histories[object_id] = deque(
                maxlen=self.max_frames
            )

        self.histories[object_id].append(prediction)

        return self.get_smoothed_prediction(object_id)

    def get_smoothed_prediction(self, object_id):
        """
        Return the most common prediction for an object.
        """

        history = self.histories.get(object_id)

        if not history:
            return None

        counts = Counter(history)

        return counts.most_common(1)[0][0]

    def reset_object(self, object_id):
        """
        Remove the history of one object.
        """

        self.histories.pop(object_id, None)

    def reset(self):
        """
        Clear all prediction histories.
        """

        self.histories.clear()

    def cleanup(self, active_object_ids):
        """
        Remove histories for objects that are no longer active.
        """

        active_object_ids = set(active_object_ids)

        inactive_ids = [
            object_id
            for object_id in self.histories
            if object_id not in active_object_ids
        ]

        for object_id in inactive_ids:
            del self.histories[object_id]

    def __len__(self):
        return len(self.histories)