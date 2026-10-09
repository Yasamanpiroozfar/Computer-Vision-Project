from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from mtcnn import MTCNN
except ModuleNotFoundError as exc:
    MTCNN = None
    _MTCNN_IMPORT_ERROR = exc
else:
    _MTCNN_IMPORT_ERROR = None


@dataclass
class FaceDetectionResult:
    image: np.ndarray
    """The image."""
    rect: tuple[int, int, int, int]
    """The face bounding box (top left x, top left y, width, height)."""
    aligned: np.ndarray
    """The aligned face image."""


# The FaceDetector class provides methods for detection, tracking, and alignment of faces.
class FaceDetector:

    # Prepare the face detector; specify all parameters used for detection, tracking, and alignment.
    def __init__(
        self, tm_window_size: int = 25, tm_threshold: float = 0.70, aligned_image_size: int = 224
    ) -> None:
        if MTCNN is None:
            raise RuntimeError(
                "MTCNN could not be imported. Install the requirements, including a compatible "
                "TensorFlow package, before using FaceDetector."
            ) from _MTCNN_IMPORT_ERROR

        # Prepare face detection and alignment.
        self.detector = MTCNN()

        self.reference: Optional[FaceDetectionResult] = None

        # Size of face image after crop-and-resize alignment.
        self.aligned_image_size = int(aligned_image_size)

        # Template matching parameters.
        self.tm_window_size = max(0, int(tm_window_size))
        self.tm_threshold = float(tm_threshold)
        self.tm_method = cv2.TM_CCOEFF_NORMED

    # Track a face in a new image using template matching.
    def track_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        if image is None or image.size == 0:
            self.reference = None
            return None

        if self.reference is None:
            detected = self.detect_face(image)
            self.reference = detected
            return detected

        ref_rect = self._clip_rect(self.reference.rect, self.reference.image.shape)
        template = self.crop_face(self.reference.image, ref_rect)
        if template.size == 0:
            detected = self.detect_face(image)
            self.reference = detected
            return detected

        x, y, width, height = ref_rect
        margin = self.tm_window_size
        left = max(0, x - margin)
        top = max(0, y - margin)
        right = min(image.shape[1], x + width + margin)
        bottom = min(image.shape[0], y + height + margin)
        search_region = image[top:bottom, left:right]

        # matchTemplate requires the search region to be at least as large as the template.
        if (
            search_region.shape[0] < template.shape[0]
            or search_region.shape[1] < template.shape[1]
        ):
            detected = self.detect_face(image)
            self.reference = detected
            return detected

        response = cv2.matchTemplate(search_region, template, self.tm_method)
        _, max_value, _, max_location = cv2.minMaxLoc(response)

        # Re-detect if tracking is lost.
        if not np.isfinite(max_value) or max_value < self.tm_threshold:
            detected = self.detect_face(image)
            self.reference = detected
            return detected

        tracked_rect = (
            left + int(max_location[0]),
            top + int(max_location[1]),
            template.shape[1],
            template.shape[0],
        )
        tracked_rect = self._clip_rect(tracked_rect, image.shape)
        aligned = self.align_face(image, tracked_rect)
        result = FaceDetectionResult(rect=tracked_rect, image=image, aligned=aligned)

        self.reference = result
        return result

    # Face detection in a new image.
    def detect_face(self, image: np.ndarray) -> Optional[FaceDetectionResult]:
        # OpenCV frames are BGR, while MTCNN expects RGB input.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not (
            detections := self.detector.detect_faces(
                rgb_image, threshold_pnet=0.85, threshold_rnet=0.9
            )
        ):
            self.reference = None
            return None

        # Select face with the largest bounding box.
        largest_detection = np.argmax([d["box"][2] * d["box"][3] for d in detections])
        face_rect = self._clip_rect(tuple(detections[largest_detection]["box"]), image.shape)
        if face_rect[2] <= 0 or face_rect[3] <= 0:
            self.reference = None
            return None

        # Align the detected face.
        aligned = self.align_face(image, face_rect)
        result = FaceDetectionResult(rect=face_rect, image=image, aligned=aligned)
        return result

    # Face alignment to predefined size.
    def align_face(self, image: np.ndarray, face_rect: tuple[int, int, int, int]) -> np.ndarray:
        cropped = self.crop_face(image, face_rect)
        if cropped.size == 0:
            raise ValueError("Cannot align an empty face crop.")
        return cv2.resize(
            cropped,
            dsize=(self.aligned_image_size, self.aligned_image_size),
            interpolation=cv2.INTER_LINEAR,
        )

    # Crop face according to detected bounding box.
    def crop_face(self, image: np.ndarray, face_rect: tuple[int, int, int, int]) -> np.ndarray:
        left, top, width, height = self._clip_rect(face_rect, image.shape)
        right = left + width
        bottom = top + height
        return image[top:bottom, left:right, :]

    @staticmethod
    def _clip_rect(
        face_rect: tuple[int, int, int, int], image_shape: tuple[int, ...]
    ) -> tuple[int, int, int, int]:
        x, y, width, height = [int(v) for v in face_rect]
        image_height, image_width = image_shape[:2]
        left = min(max(x, 0), image_width)
        top = min(max(y, 0), image_height)
        right = min(max(x + max(width, 0), 0), image_width)
        bottom = min(max(y + max(height, 0), 0), image_height)
        return left, top, max(0, right - left), max(0, bottom - top)
