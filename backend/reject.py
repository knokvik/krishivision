import cv2
import numpy as np
from typing import Tuple, Dict, Optional
from PIL import Image
import io

from config import CONFIDENCE_THRESHOLD, NON_LEAF_THRESHOLD, IMAGE_SIZE


class RejectionModule:
    """Handle non-leaf detection and low-confidence rejection."""
    
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD,
                 non_leaf_threshold: float = NON_LEAF_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.non_leaf_threshold = non_leaf_threshold
        
        self.leaf_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def is_leaf_image(self, image: np.ndarray) -> Tuple[bool, float, str]:
        """
        Detect if image contains a leaf using color and texture analysis.
        
        Returns:
            (is_leaf, confidence_score, reason)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        green_lower = np.array([35, 30, 30])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        green_ratio = np.sum(green_mask > 0) / (image.shape[0] * image.shape[1])
        
        yellow_lower = np.array([15, 30, 30])
        yellow_upper = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        yellow_ratio = np.sum(yellow_mask > 0) / (image.shape[0] * image.shape[1])
        
        brown_lower = np.array([8, 30, 30])
        brown_upper = np.array([20, 255, 200])
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        brown_ratio = np.sum(brown_mask > 0) / (image.shape[0] * image.shape[1])
        
        leaf_color_ratio = green_ratio + yellow_ratio + brown_ratio
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(laplacian_var / 500.0, 1.0)
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (image.shape[0] * image.shape[1])
        
        color_score = min(leaf_color_ratio * 2, 1.0)
        texture_score = min(edge_density * 10, 1.0)
        
        combined_score = (color_score * 0.5 + texture_score * 0.3 + blur_score * 0.2)
        
        is_leaf = combined_score > self.non_leaf_threshold
        
        if not is_leaf:
            if leaf_color_ratio < 0.1:
                reason = "Image does not appear to contain plant matter (insufficient green/yellow/brown pixels)"
            elif blur_score < 0.3:
                reason = "Image appears blurry or out of focus"
            elif edge_density < 0.01:
                reason = "Image lacks leaf-like texture and edges"
            else:
                reason = "Image does not resemble a plant leaf"
        else:
            reason = "Leaf detected"
        
        return is_leaf, combined_score, reason
    
    def check_confidence(self, confidence: float) -> Tuple[bool, str]:
        """
        Check if prediction confidence meets threshold.
        
        Returns:
            (passes_threshold, reason)
        """
        if confidence >= self.confidence_threshold:
            return True, f"Confidence {confidence:.2%} meets threshold {self.confidence_threshold:.0%}"
        else:
            return False, f"Confidence {confidence:.2%} below threshold {self.confidence_threshold:.0%}"
    
    def validate_image(self, image: np.ndarray) -> Tuple[bool, str]:
        """
        Validate image format and basic properties.
        
        Returns:
            (is_valid, reason)
        """
        if image is None:
            return False, "Image is None"
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            return False, f"Expected RGB image with 3 channels, got shape {image.shape}"
        
        if image.shape[0] < 50 or image.shape[1] < 50:
            return False, f"Image too small: {image.shape[0]}x{image.shape[1]}"
        
        if image.dtype != np.uint8:
            return False, f"Expected uint8 image, got {image.dtype}"
        
        if np.all(image == 0) or np.all(image == 255):
            return False, "Image appears to be blank (all black or all white)"
        
        return True, "Image format valid"
    
    def process_prediction(
        self,
        image: np.ndarray,
        predictions: list,
        top_prediction: dict
    ) -> Dict:
        """
        Process prediction through rejection pipeline.
        
        Args:
            image: Input image
            predictions: List of top-k predictions with confidence
            top_prediction: Top prediction dict with class, confidence
        
        Returns:
            Dictionary with verdict and details
        """
        valid, reason = self.validate_image(image)
        if not valid:
            return {
                "verdict": "rejected",
                "reason": f"Invalid image: {reason}",
                "confidence": 0.0,
                "predictions": []
            }
        
        is_leaf, leaf_score, leaf_reason = self.is_leaf_image(image)
        if not is_leaf:
            return {
                "verdict": "rejected",
                "reason": f"Non-leaf image: {leaf_reason}",
                "leaf_score": leaf_score,
                "confidence": 0.0,
                "predictions": []
            }
        
        confidence = top_prediction["confidence"]
        passes_conf, conf_reason = self.check_confidence(confidence)
        
        if not passes_conf:
            return {
                "verdict": "unsure",
                "reason": conf_reason,
                "leaf_score": leaf_score,
                "confidence": confidence,
                "predictions": predictions,
                "top_class": top_prediction["class"]
            }
        
        return {
            "verdict": "accepted",
            "reason": f"Leaf detected (score: {leaf_score:.2f}), confidence {confidence:.2%} accepted",
            "leaf_score": leaf_score,
            "confidence": confidence,
            "predictions": predictions,
            "top_class": top_prediction["class"]
        }


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load and preprocess image from bytes."""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = image.resize(IMAGE_SIZE)
    return np.array(image)


def preprocess_image_from_path(image_path: str) -> np.ndarray:
    """Load and preprocess image from file path."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, IMAGE_SIZE)
    return image


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test rejection module")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold")
    args = parser.parse_args()
    
    rejector = RejectionModule(confidence_threshold=args.confidence)
    
    image = preprocess_image_from_path(args.image)
    
    valid, reason = rejector.validate_image(image)
    print(f"Image valid: {valid} - {reason}")
    
    is_leaf, leaf_score, leaf_reason = rejector.is_leaf_image(image)
    print(f"Is leaf: {is_leaf} (score: {leaf_score:.3f}) - {leaf_reason}")
    
    fake_predictions = [
        {"class": "Tomato___healthy", "confidence": 0.85, "class_index": 37},
        {"class": "Tomato___Early_blight", "confidence": 0.10, "class_index": 29},
        {"class": "Tomato___Late_blight", "confidence": 0.05, "class_index": 30}
    ]
    
    result = rejector.process_prediction(image, fake_predictions, fake_predictions[0])
    print(f"\nVerdict: {result['verdict']}")
    print(f"Reason: {result['reason']}")