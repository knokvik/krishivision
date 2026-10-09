import pytest
import io
import base64
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.config import CLASS_NAMES, CONFIDENCE_THRESHOLD


client = TestClient(app)


@pytest.fixture
def sample_image():
    """Create a synthetic test image."""
    img = Image.new('RGB', (224, 224), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes


@pytest.fixture
def sample_image_bytes():
    """Create synthetic test image bytes."""
    img = Image.new('RGB', (224, 224), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


def test_health_endpoint():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


def test_predict_endpoint_accepts_image(sample_image_bytes):
    """Test predict endpoint accepts valid image."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/predict", files=files, data={"language": "en"})
    
    assert response.status_code == 200
    data = response.json()
    assert "verdict" in data
    assert data["verdict"] in ["accepted", "rejected", "unsure"]


def test_predict_rejects_non_image():
    """Test predict endpoint rejects non-image files."""
    files = {"file": ("test.txt", b"not an image", "text/plain")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == 422 or response.status_code == 400


def test_predict_with_language(sample_image_bytes):
    """Test predict with different languages."""
    for lang in ["en", "hi", "mr"]:
        files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
        response = client.post("/predict", files=files, data={"language": lang})
        assert response.status_code == 200


def test_remedy_endpoint():
    """Test remedy endpoint."""
    form_data = {
        "class_name": "Tomato___Early_blight",
        "confidence": "0.85"
    }
    response = client.post("/remedy?lang=en", data=form_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "crop" in data
    assert "disease" in data
    assert "organic_treatment" in data
    assert "chemical_treatment" in data
    assert "prevention" in data
    assert "urgency" in data
    assert "severity" in data
    assert "language" in data


def test_remedy_endpoint_invalid_class():
    """Test remedy endpoint with invalid class."""
    form_data = {
        "class_name": "Invalid___Class",
        "confidence": "0.85"
    }
    response = client.post("/remedy?lang=en", data=form_data)
    assert response.status_code == 400


def test_remedy_endpoint_languages():
    """Test remedy endpoint with different languages."""
    for lang in ["en", "hi", "mr"]:
        form_data = {
            "class_name": "Tomato___Early_blight",
            "confidence": "0.85"
        }
        response = client.post(f"/remedy?lang={lang}", data=form_data)
        assert response.status_code == 200
        data = response.json()
        assert data["language"] == lang


def test_report_endpoint(sample_image_bytes):
    """Test PDF report generation."""
    files = {"file": ("test.jpg", sample_image_bytes, "image/jpeg")}
    response = client.post("/report", files=files, data={"language": "en"})
    
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"


class TestOfflineFallback:
    """Integration tests with LLM forced offline."""
    
    def test_remedy_offline_mode(self, monkeypatch):
        """Test remedy generation works offline with dictionary fallback."""
        from backend.remedies.llm_remedy import get_remedy
        
        remedy = get_remedy("Tomato___Early_blight", 0.85, "en", force_offline=True)
        
        assert remedy["crop"] == "Tomato"
        assert remedy["disease"] == "Early Blight"
        assert "organic_treatment" in remedy
        assert "chemical_treatment" in remedy
        assert "prevention" in remedy
        assert remedy["language"] == "en"
    
    def test_remedy_offline_hindi(self, monkeypatch):
        """Test remedy generation in Hindi offline."""
        from backend.remedies.llm_remedy import get_remedy
        
        remedy = get_remedy("Tomato___Early_blight", 0.85, "hi", force_offline=True)
        
        assert remedy["language"] == "hi"
        assert "जैविक उपचार" in remedy["organic_treatment"] or "organic" in remedy["organic_treatment"].lower()
    
    def test_remedy_offline_marathi(self, monkeypatch):
        """Test remedy generation in Marathi offline."""
        from backend.remedies.llm_remedy import get_remedy
        
        remedy = get_remedy("Tomato___Early_blight", 0.85, "mr", force_offline=True)
        
        assert remedy["language"] == "mr"
    
    def test_remedy_healthy_class_offline(self):
        """Test remedy for healthy class offline."""
        from backend.remedies.llm_remedy import get_remedy
        
        remedy = get_remedy("Tomato___healthy", 0.95, "en", force_offline=True)
        
        assert remedy["urgency"] == "none"
        assert remedy["disease"] == "Healthy"


class TestRejectionModule:
    """Test rejection logic."""
    
    def test_reject_blank_image(self):
        """Test rejection of blank images."""
        from backend.reject import RejectionModule
        
        rejector = RejectionModule()
        blank_image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        valid, reason = rejector.validate_image(blank_image)
        assert not valid
    
    def test_reject_all_white_image(self):
        """Test rejection of all-white images."""
        from backend.reject import RejectionModule
        
        rejector = RejectionModule()
        white_image = np.full((224, 224, 3), 255, dtype=np.uint8)
        
        valid, reason = rejector.validate_image(white_image)
        assert not valid
    
    def test_reject_small_image(self):
        """Test rejection of too-small images."""
        from backend.reject import RejectionModule
        
        rejector = RejectionModule()
        small_image = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
        
        valid, reason = rejector.validate_image(small_image)
        assert not valid
    
    def test_confidence_threshold(self):
        """Test confidence threshold rejection."""
        from backend.reject import RejectionModule
        
        rejector = RejectionModule(confidence_threshold=0.7)
        
        passes, reason = rejector.check_confidence(0.8)
        assert passes
        
        passes, reason = rejector.check_confidence(0.5)
        assert not passes


class TestGradCAM:
    """Test Grad-CAM generation (requires model)."""
    
    @pytest.mark.skipif(
        not __import__("pathlib").Path("backend/ml/model.keras").exists(),
        reason="Model not available"
    )
    def test_gradcam_generation(self):
        """Test Grad-CAM heatmap generation."""
        from backend.gradcam import create_gradcam_generator
        import cv2
        
        generator = create_gradcam_generator("backend/ml/model.keras")
        
        test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = generator.predict_and_explain(test_image)
        
        assert "predictions" in result
        assert "heatmap_b64" in result
        assert "overlay_b64" in result
        assert len(result["predictions"]) == 3
        
        heatmap = base64.b64decode(result["heatmap_b64"])
        assert len(heatmap) > 0


class TestPDFReport:
    """Test PDF report generation."""
    
    def test_pdf_generation(self, tmp_path):
        """Test PDF report generation."""
        from backend.pdf_report import generate_pdf_report
        
        prediction = {
            "crop": "Tomato",
            "disease": "Early Blight",
            "confidence": 0.87,
            "verdict": "accepted",
            "top_3": [
                {"class": "Tomato___Early_blight", "confidence": 0.87},
                {"class": "Tomato___Late_blight", "confidence": 0.08},
                {"class": "Tomato___Septoria_leaf_spot", "confidence": 0.03}
            ]
        }
        
        remedy = {
            "crop": "Tomato",
            "disease": "Early Blight",
            "organic_treatment": "Test organic treatment",
            "chemical_treatment": "Test chemical treatment",
            "prevention": "Test prevention",
            "urgency": "medium",
            "severity": "Test severity",
            "language": "en"
        }
        
        output_path = tmp_path / "test_report.pdf"
        result_path = generate_pdf_report(prediction, remedy, output_path=str(output_path))
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])