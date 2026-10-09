import pytest
import numpy as np
from PIL import Image
import io

from backend.reject import RejectionModule, preprocess_image_from_bytes


class TestNonLeafDetection:
    """Test non-leaf image detection."""
    
    def setup_method(self):
        self.rejector = RejectionModule()
    
    def test_detect_green_leaf(self):
        """Test detection of green leaf-like image."""
        green_image = np.zeros((224, 224, 3), dtype=np.uint8)
        green_image[:, :, 1] = 150
        
        is_leaf, score, reason = self.rejector.is_leaf_image(green_image)
        assert is_leaf
        assert score > 0.3
    
    def test_detect_yellow_leaf(self):
        """Test detection of yellowing leaf."""
        yellow_image = np.zeros((224, 224, 3), dtype=np.uint8)
        yellow_image[:, :, 0] = 200
        yellow_image[:, :, 1] = 180
        
        is_leaf, score, reason = self.rejector.is_leaf_image(yellow_image)
        assert is_leaf
    
    def test_detect_brown_leaf(self):
        """Test detection of brown leaf."""
        brown_image = np.zeros((224, 224, 3), dtype=np.uint8)
        brown_image[:, :, 0] = 100
        brown_image[:, :, 1] = 60
        brown_image[:, :, 2] = 30
        
        is_leaf, score, reason = self.rejector.is_leaf_image(brown_image)
        assert is_leaf
    
    def test_reject_blue_image(self):
        """Test rejection of blue (non-leaf) image."""
        blue_image = np.zeros((224, 224, 3), dtype=np.uint8)
        blue_image[:, :, 2] = 200
        
        is_leaf, score, reason = self.rejector.is_leaf_image(blue_image)
        assert not is_leaf
        assert "plant matter" in reason.lower() or "green" in reason.lower()
    
    def test_reject_red_image(self):
        """Test rejection of red (non-leaf) image."""
        red_image = np.zeros((224, 224, 3), dtype=np.uint8)
        red_image[:, :, 0] = 200
        
        is_leaf, score, reason = self.rejector.is_leaf_image(red_image)
        assert not is_leaf
    
    def test_reject_grayscale_image(self):
        """Test rejection of grayscale image."""
        gray_image = np.full((224, 224, 3), 128, dtype=np.uint8)
        
        is_leaf, score, reason = self.rejector.is_leaf_image(gray_image)
        assert not is_leaf


class TestBlurDetection:
    """Test blur detection in rejection."""
    
    def setup_method(self):
        self.rejector = RejectionModule()
    
    def test_sharp_image_passes(self):
        """Test sharp image passes blur check."""
        sharp_image = np.zeros((224, 224, 3), dtype=np.uint8)
        sharp_image[::2, ::2] = [0, 255, 0]
        sharp_image[1::2, 1::2] = [0, 100, 0]
        
        is_leaf, score, reason = self.rejector.is_leaf_image(sharp_image)
        assert is_leaf
    
    def test_blurry_image_rejected(self):
        """Test blurry image gets low score."""
        blurry_image = np.full((224, 224, 3), 100, dtype=np.uint8)
        noise = np.random.normal(0, 5, (224, 224, 3))
        blurry_image = np.clip(blurry_image + noise, 0, 255).astype(np.uint8)
        
        is_leaf, score, reason = self.rejector.is_leaf_image(blurry_image)
        assert score < 0.5


class TestCorruptedInputs:
    """Test handling of corrupted/invalid inputs."""
    
    def setup_method(self):
        self.rejector = RejectionModule()
    
    def test_none_image(self):
        """Test None image is rejected."""
        valid, reason = self.rejector.validate_image(None)
        assert not valid
    
    def test_wrong_channels(self):
        """Test wrong number of channels."""
        rgba_image = np.random.randint(0, 255, (224, 224, 4), dtype=np.uint8)
        valid, reason = self.rejector.validate_image(rgba_image)
        assert not valid
    
    def test_grayscale_2d(self):
        """Test 2D grayscale image."""
        gray_image = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        valid, reason = self.rejector.validate_image(gray_image)
        assert not valid
    
    def test_wrong_dtype(self):
        """Test float image instead of uint8."""
        float_image = np.random.rand(224, 224, 3).astype(np.float32)
        valid, reason = self.rejector.validate_image(float_image)
        assert not valid
    
    def test_empty_bytes(self):
        """Test empty bytes preprocessing."""
        with pytest.raises(Exception):
            preprocess_image_from_bytes(b"")
    
    def test_corrupted_jpeg(self):
        """Test corrupted JPEG bytes."""
        with pytest.raises(Exception):
            preprocess_image_from_bytes(b"not a valid image")


class TestConfidenceThreshold:
    """Test confidence threshold logic."""
    
    def test_default_threshold(self):
        """Test default 70% threshold."""
        rejector = RejectionModule()
        
        passes, _ = rejector.check_confidence(0.70)
        assert passes
        
        passes, _ = rejector.check_confidence(0.69)
        assert not passes
    
    def test_custom_threshold(self):
        """Test custom threshold."""
        rejector = RejectionModule(confidence_threshold=0.5)
        
        passes, _ = rejector.check_confidence(0.5)
        assert passes
        
        passes, _ = rejector.check_confidence(0.49)
        assert not passes
    
    def test_high_confidence_passes(self):
        """Test very high confidence passes."""
        rejector = RejectionModule()
        
        passes, reason = rejector.check_confidence(0.99)
        assert passes
        assert "99%" in reason


class TestFullPipeline:
    """Test full rejection pipeline with mock predictions."""
    
    def setup_method(self):
        self.rejector = RejectionModule()
    
    def test_accepted_pipeline(self):
        """Test full pipeline accepts good leaf with high confidence."""
        green_image = np.zeros((224, 224, 3), dtype=np.uint8)
        green_image[:, :, 1] = 150
        
        predictions = [
            {"class": "Tomato___healthy", "confidence": 0.9, "class_index": 37},
            {"class": "Tomato___Early_blight", "confidence": 0.07, "class_index": 29},
            {"class": "Tomato___Late_blight", "confidence": 0.03, "class_index": 30}
        ]
        
        result = self.rejector.process_prediction(green_image, predictions, predictions[0])
        
        assert result["verdict"] == "accepted"
        assert result["confidence"] == 0.9
        assert result["top_class"] == "Tomato___healthy"
    
    def test_unsure_pipeline(self):
        """Test pipeline returns unsure for low confidence."""
        green_image = np.zeros((224, 224, 3), dtype=np.uint8)
        green_image[:, :, 1] = 150
        
        predictions = [
            {"class": "Tomato___Early_blight", "confidence": 0.55, "class_index": 29},
            {"class": "Tomato___Late_blight", "confidence": 0.3, "class_index": 30},
            {"class": "Tomato___healthy", "confidence": 0.15, "class_index": 37}
        ]
        
        result = self.rejector.process_prediction(green_image, predictions, predictions[0])
        
        assert result["verdict"] == "unsure"
        assert result["confidence"] == 0.55
    
    def test_rejected_non_leaf(self):
        """Test pipeline rejects non-leaf."""
        blue_image = np.zeros((224, 224, 3), dtype=np.uint8)
        blue_image[:, :, 2] = 200
        
        predictions = [
            {"class": "Tomato___healthy", "confidence": 0.9, "class_index": 37}
        ]
        
        result = self.rejector.process_prediction(blue_image, predictions, predictions[0])
        
        assert result["verdict"] == "rejected"
        assert "Non-leaf" in result["reason"]
    
    def test_rejected_invalid_image(self):
        """Test pipeline rejects invalid image."""
        invalid_image = None
        
        predictions = [
            {"class": "Tomato___healthy", "confidence": 0.9, "class_index": 37}
        ]
        
        result = self.rejector.process_prediction(invalid_image, predictions, predictions[0])
        
        assert result["verdict"] == "rejected"
        assert "Invalid image" in result["reason"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])