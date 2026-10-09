import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras import models
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils import normalize
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from config import IMAGE_SIZE, CLASS_NAMES, MODEL_KERAS_PATH


class GradCAMGenerator:
    """Generate Grad-CAM heatmaps for model predictions."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(MODEL_KERAS_PATH)
        self.model = None
        self.gradcam = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and setup Grad-CAM."""
        try:
            self.model = models.load_model(self.model_path)
            self.gradcam = Gradcam(
                self.model,
                model_modifier=ReplaceToLinear(),
                clone=True
            )
            print(f"Model loaded from {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> tf.Tensor:
        """Preprocess image for model input."""
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.expand_dims(image, axis=0)
        return image
    
    def _get_last_conv_layer(self) -> str:
        """Find the last convolutional layer name."""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
            if hasattr(layer, 'layers'):
                for sublayer in reversed(layer.layers):
                    if isinstance(sublayer, tf.keras.layers.Conv2D):
                        return sublayer.name
        return 'block_16_project_BN'
    
    def generate_heatmap(
        self,
        image: np.ndarray,
        class_index: int = None,
        penultimate_layer: str = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a given image and class.
        
        Args:
            image: Input image as numpy array (H, W, 3) in range [0, 1] or [0, 255]
            class_index: Target class index. If None, uses predicted class.
            penultimate_layer: Name of the layer to use for Grad-CAM.
        
        Returns:
            Heatmap as numpy array (H, W) in range [0, 1]
        """
        if penultimate_layer is None:
            penultimate_layer = self._get_last_conv_layer()
        
        input_tensor = self._preprocess_image(image)
        
        if class_index is None:
            predictions = self.model(input_tensor)
            class_index = tf.argmax(predictions[0]).numpy()
        
        def score_function(output):
            return output[:, class_index]
        
        cam = self.gradcam(
            score_function,
            input_tensor,
            penultimate_layer=penultimate_layer
        )
        
        cam = normalize(cam)
        heatmap = cam[0].numpy()
        
        return heatmap
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            image: Original image (H, W, 3) in range [0, 255]
            heatmap: Grad-CAM heatmap (H, W) in range [0, 1]
            alpha: Blending factor for heatmap overlay
            colormap: OpenCV colormap to use
        
        Returns:
            Overlaid image as numpy array (H, W, 3) in range [0, 255]
        """
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay
    
    def heatmap_to_base64(self, heatmap: np.ndarray, format: str = 'PNG') -> str:
        """Convert heatmap to base64 encoded string."""
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(heatmap_colored)
        buffered = BytesIO()
        pil_img.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def overlay_to_base64(self, overlay: np.ndarray, format: str = 'PNG') -> str:
        """Convert overlay image to base64 encoded string."""
        pil_img = Image.fromarray(overlay.astype(np.uint8))
        buffered = BytesIO()
        pil_img.save(buffered, format=format)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def predict_and_explain(
        self,
        image: np.ndarray,
        top_k: int = 3
    ) -> dict:
        """
        Run prediction and generate Grad-CAM for top prediction.
        
        Returns:
            Dictionary with predictions, heatmap, and overlay
        """
        input_tensor = self._preprocess_image(image)
        predictions = self.model(input_tensor, training=False).numpy()[0]
        
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_predictions = [
            {
                "class": CLASS_NAMES[idx],
                "confidence": float(predictions[idx]),
                "class_index": int(idx)
            }
            for idx in top_indices
        ]
        
        top_class_idx = top_indices[0]
        heatmap = self.generate_heatmap(image, class_index=top_class_idx)
        overlay = self.overlay_heatmap(image, heatmap)
        
        return {
            "predictions": top_predictions,
            "heatmap_b64": self.heatmap_to_base64(heatmap),
            "overlay_b64": self.overlay_to_base64(overlay),
            "top_class": CLASS_NAMES[top_class_idx],
            "top_confidence": float(predictions[top_class_idx])
        }


def create_gradcam_generator(model_path: str = None) -> GradCAMGenerator:
    """Factory function to create GradCAMGenerator."""
    return GradCAMGenerator(model_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Grad-CAM generation")
    parser.add_argument("--model", type=str, help="Path to model")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--output", type=str, default="gradcam_output", help="Output directory")
    args = parser.parse_args()
    
    from pathlib import Path
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = create_gradcam_generator(args.model)
    
    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result = generator.predict_and_explain(image)
    
    print(f"Top prediction: {result['top_class']} ({result['top_confidence']:.4f})")
    print(f"Top 3 predictions:")
    for pred in result['predictions']:
        print(f"  {pred['class']}: {pred['confidence']:.4f}")
    
    heatmap_img = base64.b64decode(result['heatmap_b64'])
    overlay_img = base64.b64decode(result['overlay_b64'])
    
    with open(output_dir / "heatmap.png", 'wb') as f:
        f.write(heatmap_img)
    with open(output_dir / "overlay.png", 'wb') as f:
        f.write(overlay_img)
    
    print(f"Output saved to {output_dir}/")