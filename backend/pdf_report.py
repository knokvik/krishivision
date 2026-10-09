import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib import colors

from config import CLASS_NAMES, MODEL_VERSION, LANGUAGES


class PDFReportGenerator:
    """Generate diagnosis PDF reports using ReportLab."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='TitleStyle',
            parent=self.styles['Title'],
            fontSize=24,
            leading=28,
            textColor=HexColor('#2E7D32'),
            spaceAfter=6,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubtitleStyle',
            parent=self.styles['Normal'],
            fontSize=14,
            leading=18,
            textColor=HexColor('#555555'),
            spaceAfter=12,
            alignment=TA_CENTER
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            leading=18,
            textColor=HexColor('#1B5E20'),
            spaceBefore=12,
            spaceAfter=6,
            borderWidth=0,
            borderPadding=0,
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyStyle',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=14,
            textColor=HexColor('#333333'),
            spaceAfter=6,
            alignment=TA_JUSTIFY
        ))
        
        self.styles.add(ParagraphStyle(
            name='SmallStyle',
            parent=self.styles['Normal'],
            fontSize=8,
            leading=10,
            textColor=HexColor('#777777'),
            spaceAfter=4,
        ))
    
    def _decode_base64_image(self, b64_string: str) -> Optional[Image.Image]:
        """Decode base64 string to PIL Image."""
        try:
            img_data = base64.b64decode(b64_string)
            return Image.open(io.BytesIO(img_data))
        except Exception as e:
            print(f"Error decoding image: {e}")
            return None
    
    def _save_temp_image(self, pil_image: Image.Image, prefix: str = "temp") -> Optional[str]:
        """Save PIL image to temporary file and return path."""
        try:
            temp_path = f"/tmp/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            pil_image.save(temp_path, format='PNG')
            return temp_path
        except Exception as e:
            print(f"Error saving temp image: {e}")
            return None
    
    def _create_diagnosis_table(self, prediction: Dict) -> Table:
        """Create diagnosis summary table."""
        data = [
            ['Crop', prediction.get('crop', 'Unknown')],
            ['Disease', prediction.get('disease', 'Unknown')],
            ['Confidence', f"{prediction.get('confidence', 0):.1%}"],
            ['Status', prediction.get('verdict', 'Unknown').capitalize()],
        ]
        
        if 'top_3' in prediction:
            top3_str = ", ".join([
                f"{p['class'].replace('___', ' - ')} ({p['confidence']:.1%})"
                for p in prediction['top_3']
            ])
            data.append(['Top 3 Predictions', top3_str])
        
        table = Table(data, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#E8F5E9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#333333')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        return table
    
    def _create_remedy_table(self, remedy: Dict) -> Table:
        """Create remedy details table."""
        data = [
            ['Treatment Type', 'Details'],
            ['Organic', remedy.get('organic_treatment', 'N/A')],
            ['Chemical', remedy.get('chemical_treatment', 'N/A')],
            ['Prevention', remedy.get('prevention', 'N/A')],
            ['Urgency', remedy.get('urgency', 'N/A').upper()],
            ['Severity', remedy.get('severity', 'N/A')],
        ]
        
        table = Table(data, colWidths=[1.5*inch, 5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#2E7D32')),
            ('TEXTCOLOR', (0, 0), (-1, 0), white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, HexColor('#F5F5F5')]),
        ]))
        return table
    
    def generate_report(
        self,
        prediction: Dict,
        remedy: Dict,
        heatmap_b64: str = None,
        overlay_b64: str = None,
        original_image_b64: str = None,
        language: str = 'en',
        output_path: str = None
    ) -> str:
        """
        Generate PDF report.
        
        Args:
            prediction: Prediction results dict
            remedy: Remedy dict from llm_remedy
            heatmap_b64: Base64 encoded Grad-CAM heatmap
            overlay_b64: Base64 encoded overlay image
            original_image_b64: Base64 encoded original image
            language: Language code
            output_path: Output file path (auto-generated if None)
        
        Returns:
            Path to generated PDF
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"krishivision_report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )
        
        story = []
        
        story.append(Paragraph("KrishiVision", self.styles['TitleStyle']))
        story.append(Paragraph("AI Crop Disease Diagnosis Report", self.styles['SubtitleStyle']))
        story.append(Spacer(1, 12))
        
        meta_data = [
            ['Report Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Model Version', MODEL_VERSION],
            ['Language', LANGUAGES.get(language, language)],
        ]
        meta_table = Table(meta_data, colWidths=[2*inch, 4.5*inch])
        meta_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#555555')),
            ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#EEEEEE')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 16))
        
        story.append(Paragraph("Diagnosis", self.styles['SectionHeader']))
        story.append(self._create_diagnosis_table(prediction))
        story.append(Spacer(1, 16))
        
        if overlay_b64:
            story.append(Paragraph("Grad-CAM Heatmap Overlay", self.styles['SectionHeader']))
            overlay_img = self._decode_base64_image(overlay_b64)
            if overlay_img:
                temp_path = self._save_temp_image(overlay_img, "overlay")
                if temp_path:
                    rl_image = RLImage(temp_path, width=5*inch, height=3.5*inch)
                    story.append(rl_image)
                    story.append(Paragraph(
                        "Regions in red/yellow indicate areas that most influenced the diagnosis.",
                        self.styles['SmallStyle']
                    ))
        
        story.append(Spacer(1, 16))
        story.append(Paragraph("Treatment Recommendations", self.styles['SectionHeader']))
        story.append(self._create_remedy_table(remedy))
        story.append(Spacer(1, 20))
        
        disclaimer = """
        <b>Disclaimer:</b> This report is generated by an AI system for informational purposes only. 
        It is not a substitute for professional agricultural advice. Please consult with local 
        agricultural extension officers, certified crop advisors, or plant pathologists for 
        definitive diagnosis and treatment recommendations. The model may produce false positives 
        or negatives. Always verify with field observations and laboratory testing when possible.
        """
        story.append(Paragraph(disclaimer, self.styles['SmallStyle']))
        story.append(Spacer(1, 12))
        
        footer = f"Generated by KrishiVision v{MODEL_VERSION} | {datetime.now().strftime('%Y-%m-%d')}"
        story.append(Paragraph(footer, self.styles['SmallStyle']))
        
        doc.build(story)
        
        import tempfile
        for f in Path("/tmp").glob("temp_*.png"):
            try:
                f.unlink()
            except:
                pass
        
        return output_path


def generate_pdf_report(
    prediction: Dict,
    remedy: Dict,
    heatmap_b64: str = None,
    overlay_b64: str = None,
    original_image_b64: str = None,
    language: str = 'en',
    output_path: str = None
) -> str:
    """Convenience function to generate PDF report."""
    generator = PDFReportGenerator()
    return generator.generate_report(
        prediction, remedy, heatmap_b64, overlay_b64,
        original_image_b64, language, output_path
    )


if __name__ == "__main__":
    test_prediction = {
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
    
    test_remedy = {
        "crop": "Tomato",
        "disease": "Early Blight",
        "organic_treatment": "Mulch around plants. Stake and prune for air circulation. Apply copper or sulfur fungicides weekly. Remove lower infected leaves.",
        "chemical_treatment": "Chlorothalonil, mancozeb, or azoxystrobin. Start at first fruit set. Apply every 7-10 days. Rotate modes of action.",
        "prevention": "Crop rotation (3+ years). Mulch. Staking. Resistant varieties. Balanced fertility. Avoid overhead irrigation.",
        "urgency": "medium",
        "severity": "Defoliation, fruit sunscald, yield loss if untreated",
        "language": "en"
    }
    
    output = generate_pdf_report(test_prediction, test_remedy)
    print(f"Test report generated: {output}")