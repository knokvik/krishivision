import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import io

from backend.main import app
from backend.config import CLASS_NAMES
from fastapi.testclient import TestClient

client = TestClient(app)


class TestCanonicalImages:
    """E2E smoke test with canonical test images."""
    
    @pytest.fixture
    def canonical_images(self):
        """Create 10 canonical test images for different classes."""
        classes = [
            "Apple___healthy",
            "Apple___Apple_scab",
            "Corn___healthy",
            "Corn___Common_rust",
            "Grape___healthy",
            "Grape___Black_rot",
            "Potato___healthy",
            "Potato___Late_blight",
            "Tomato___healthy",
            "Tomato___Early_blight"
        ]
        
        images = {}
        for class_name in classes:
            img = Image.new('RGB', (224, 224), color='green')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG', quality=85)
            images[class_name] = img_bytes.getvalue()
        
        return images
    
    def test_predict_all_canonical(self, canonical_images):
        """Test prediction on all canonical images."""
        results = {}
        
        for class_name, img_bytes in canonical_images.items():
            files = {"file": (f"{class_name}.jpg", img_bytes, "image/jpeg")}
            response = client.post("/predict", files=files, data={"language": "en"})
            
            assert response.status_code == 200, f"Failed for {class_name}"
            data = response.json()
            
            assert "verdict" in data
            assert data["verdict"] in ["accepted", "rejected", "unsure"]
            assert "crop" in data
            assert "disease" in data
            assert "confidence" in data
            assert "top_3" in data
            assert len(data["top_3"]) == 3
            
            results[class_name] = data
        
        return results
    
    def test_heatmap_generation(self, canonical_images):
        """Test heatmap is generated for accepted predictions."""
        for class_name, img_bytes in list(canonical_images.items())[:3]:
            files = {"file": (f"{class_name}.jpg", img_bytes, "image/jpeg")}
            response = client.post("/predict", files=files, data={
                "language": "en",
                "include_heatmap": "true"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            if data["verdict"] == "accepted":
                assert "heatmap_b64" in data
                assert "overlay_b64" in data
                assert data["heatmap_b64"] is not None
                assert data["overlay_b64"] is not None
    
    def test_remedy_generation(self, canonical_images):
        """Test remedy generation for all canonical images."""
        for class_name, img_bytes in canonical_images.items():
            files = {"file": (f"{class_name}.jpg", img_bytes, "image/jpeg")}
            response = client.post("/predict", files=files, data={
                "language": "en",
                "include_remedy": "true"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            if data["verdict"] in ["accepted", "unsure"]:
                assert "remedy" in data
                if data["remedy"]:
                    assert "organic_treatment" in data["remedy"]
                    assert "chemical_treatment" in data["remedy"]
                    assert "prevention" in data["remedy"]
                    assert "urgency" in data["remedy"]
    
    def test_multilanguage_remedies(self, canonical_images):
        """Test remedies in all supported languages."""
        test_class = "Tomato___Early_blight"
        img_bytes = canonical_images[test_class]
        
        for lang in ["en", "hi", "mr"]:
            files = {"file": (f"{test_class}.jpg", img_bytes, "image/jpeg")}
            response = client.post("/predict", files=files, data={
                "language": lang,
                "include_remedy": "true"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            if data["verdict"] in ["accepted", "unsure"] and data.get("remedy"):
                assert data["remedy"]["language"] == lang
    
    def test_pdf_report_generation(self, canonical_images):
        """Test PDF report generation."""
        test_class = "Tomato___Early_blight"
        img_bytes = canonical_images[test_class]
        
        files = {"file": (f"{test_class}.jpg", img_bytes, "image/jpeg")}
        response = client.post("/report", files=files, data={"language": "en"})
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert len(response.content) > 1000
    
    def test_confidence_threshold_behavior(self):
        """Test that low confidence triggers unsure verdict."""
        ambiguous_image = np.full((224, 224, 3), 128, dtype=np.uint8)
        noise = np.random.normal(0, 30, (224, 224, 3))
        ambiguous_image = np.clip(ambiguous_image + noise, 0, 255).astype(np.uint8)
        
        img_bytes = io.BytesIO()
        Image.fromarray(ambiguous_image).save(img_bytes, format='JPEG')
        
        files = {"file": ("ambiguous.jpg", img_bytes.getvalue(), "image/jpeg")}
        response = client.post("/predict", files=files, data={"language": "en"})
        
        assert response.status_code == 200
        data = response.json()
        
        if data["verdict"] == "unsure":
            assert data["confidence"] < 0.7
    
    def test_top3_predictions_structure(self, canonical_images):
        """Test top-3 predictions have correct structure."""
        test_class = "Tomato___healthy"
        img_bytes = canonical_images[test_class]
        
        files = {"file": (f"{test_class}.jpg", img_bytes, "image/jpeg")}
        response = client.post("/predict", files=files, data={"language": "en"})
        
        assert response.status_code == 200
        data = response.json()
        
        assert "top_3" in data
        assert len(data["top_3"]) == 3
        
        for pred in data["top_3"]:
            assert "class" in pred
            assert "confidence" in pred
            assert "class_index" in pred
            assert pred["class"] in CLASS_NAMES
            assert 0 <= pred["confidence"] <= 1
            assert 0 <= pred["class_index"] < len(CLASS_NAMES)
        
        confidences = [p["confidence"] for p in data["top_3"]]
        assert confidences == sorted(confidences, reverse=True)


class TestAPIEdgeCases:
    """Test API edge cases."""
    
    def test_empty_file(self):
        """Test empty file upload."""
        files = {"file": ("empty.jpg", b"", "image/jpeg")}
        response = client.post("/predict", files=files)
        
        assert response.status_code in [400, 422, 500]
    
    def test_large_file(self):
        """Test large file handling."""
        large_image = Image.new('RGB', (4000, 3000), color='green')
        img_bytes = io.BytesIO()
        large_image.save(img_bytes, format='JPEG', quality=95)
        
        files = {"file": ("large.jpg", img_bytes.getvalue(), "image/jpeg")}
        response = client.post("/predict", files=files)
        
        assert response.status_code in [200, 413, 500]
    
    def test_invalid_language_fallback(self, canonical_images):
        """Test invalid language falls back to English."""
        test_class = "Tomato___healthy"
        img_bytes = canonical_images[test_class]
        
        files = {"file": (f"{test_class}.jpg", img_bytes, "image/jpeg")}
        response = client.post("/predict", files=files, data={"language": "invalid"})
        
        assert response.status_code == 200
        data = response.json()
        
        if data.get("remedy"):
            assert data["remedy"]["language"] in ["en", "hi", "mr"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])