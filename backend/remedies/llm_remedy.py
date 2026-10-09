import os
import json
import requests
from typing import Dict, Optional
from pathlib import Path
from jsonschema import validate, ValidationError

from config import (
    REMEDY_DICT_PATH, MISTRAL_API_KEY, MISTRAL_API_URL,
    MISTRAL_MODEL, LANGUAGES, DEFAULT_LANGUAGE
)


REMEDY_SCHEMA = {
    "type": "object",
    "required": ["crop", "disease", "organic_treatment", "chemical_treatment", "prevention", "urgency", "severity", "language"],
    "properties": {
        "crop": {"type": "string"},
        "disease": {"type": "string"},
        "organic_treatment": {"type": "string"},
        "chemical_treatment": {"type": "string"},
        "prevention": {"type": "string"},
        "urgency": {"type": "string", "enum": ["none", "low", "medium", "high", "critical"]},
        "severity": {"type": "string"},
        "language": {"type": "string", "enum": ["en", "hi", "mr"]}
    }
}


class RemedyDictionary:
    """Local remedy dictionary fallback."""
    
    def __init__(self, dict_path: Path = REMEDY_DICT_PATH):
        self.dict_path = dict_path
        self.data = self._load()
    
    def _load(self) -> Dict:
        if self.dict_path.exists():
            with open(self.dict_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def get(self, class_name: str, language: str = DEFAULT_LANGUAGE) -> Optional[Dict]:
        entry = self.data.get(class_name)
        if not entry:
            return None
        
        return self._format_entry(entry, language)
    
    def _format_entry(self, entry: Dict, language: str) -> Dict:
        base = {
            "crop": entry.get("crop", "Unknown"),
            "disease": entry.get("disease", "Unknown"),
            "organic_treatment": entry.get("organic", ""),
            "chemical_treatment": entry.get("chemical", ""),
            "prevention": entry.get("prevention", ""),
            "urgency": entry.get("urgency", "medium"),
            "severity": entry.get("severity", ""),
            "language": language
        }
        
        if language != "en":
            base = self._translate_entry(base, language)
        
        return base
    
    def _translate_entry(self, entry: Dict, language: str) -> Dict:
        translations = {
            "hi": {
                "organic_treatment": "जैविक उपचार: " + entry["organic_treatment"],
                "chemical_treatment": "रासायनिक उपचार: " + entry["chemical_treatment"],
                "prevention": "रोकथाम: " + entry["prevention"],
                "severity": "गंभीरता: " + entry["severity"]
            },
            "mr": {
                "organic_treatment": "सेंद्रिय उपचार: " + entry["organic_treatment"],
                "chemical_treatment": "रासायनिक उपचार: " + entry["chemical_treatment"],
                "prevention": "प्रतिबंध: " + entry["prevention"],
                "severity": "तीव्रता: " + entry["severity"]
            }
        }
        
        if language in translations:
            for key, value in translations[language].items():
                entry[key] = value
        
        return entry


class LLMRemedyGenerator:
    """Generate remedies using Mistral LLM with dictionary fallback."""
    
    def __init__(self, api_key: str = MISTRAL_API_KEY):
        self.api_key = api_key
        self.dictionary = RemedyDictionary()
    
    def _build_prompt(
        self,
        crop: str,
        disease: str,
        confidence: float,
        severity: str,
        language: str
    ) -> str:
        lang_name = LANGUAGES.get(language, "English")
        
        prompt = f"""You are an agricultural expert. Provide structured treatment advice for a crop disease.

Crop: {crop}
Disease: {disease}
Confidence: {confidence:.1%}
Severity: {severity}
Language: {lang_name} ({language})

Return ONLY valid JSON matching this schema:
{{
  "crop": "string",
  "disease": "string",
  "organic_treatment": "string - organic/biological control methods",
  "chemical_treatment": "string - chemical control options with timing",
  "prevention": "string - preventive measures",
  "urgency": "none|low|medium|high|critical",
  "severity": "string - description of damage potential",
  "language": "{language}"
}}

Guidelines:
- Be specific to the crop and disease
- Include application timing for chemicals
- Mention resistance management
- Keep language appropriate for farmers
- If disease is "Healthy", return maintenance advice only
"""
        return prompt
    
    def _call_mistral(self, prompt: str) -> Optional[Dict]:
        if not self.api_key or self.api_key == "your_mistral_api_key_here":
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MISTRAL_MODEL,
            "messages": [
                {"role": "system", "content": "You are an agricultural expert providing structured crop disease remedies."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                MISTRAL_API_URL,
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"Mistral API error: {e}")
            return None
    
    def get_remedy(
        self,
        class_name: str,
        confidence: float,
        language: str = DEFAULT_LANGUAGE
    ) -> Dict:
        """
        Get remedy for a class, trying LLM first then falling back to dictionary.
        """
        crop_disease = class_name.split("___")
        crop = crop_disease[0].replace("_", " ")
        disease = crop_disease[1].replace("_", " ") if len(crop_disease) > 1 else "Unknown"
        
        is_healthy = "healthy" in disease.lower()
        severity = "No disease detected" if is_healthy else "Moderate to severe crop damage possible"
        
        llm_result = self._call_mistral(
            self._build_prompt(crop, disease, confidence, severity, language)
        )
        
        if llm_result:
            try:
                validate(instance=llm_result, schema=REMEDY_SCHEMA)
                return llm_result
            except ValidationError as e:
                print(f"LLM response validation failed: {e}")
        
        dict_result = self.dictionary.get(class_name, language)
        if dict_result:
            return dict_result
        
        return self._default_remedy(crop, disease, confidence, language)
    
    def _default_remedy(self, crop: str, disease: str, confidence: float, language: str) -> Dict:
        is_healthy = "healthy" in disease.lower()
        
        if is_healthy:
            return {
                "crop": crop,
                "disease": "Healthy",
                "organic_treatment": "No treatment needed. Continue good agricultural practices.",
                "chemical_treatment": "No treatment needed.",
                "prevention": "Maintain regular monitoring, balanced nutrition, and proper irrigation.",
                "urgency": "none",
                "severity": "No disease present",
                "language": language
            }
        
        return {
            "crop": crop,
            "disease": disease,
            "organic_treatment": "Consult local agricultural extension for organic options.",
            "chemical_treatment": "Consult local agricultural extension for approved chemicals.",
            "prevention": "Practice crop rotation, use resistant varieties, maintain field sanitation.",
            "urgency": "medium",
            "severity": "Consult expert for assessment",
            "language": language
        }


def get_remedy(
    class_name: str,
    confidence: float,
    language: str = DEFAULT_LANGUAGE,
    force_offline: bool = False
) -> Dict:
    """
    Main entry point for getting remedies.
    
    Args:
        class_name: Full class name (e.g., "Tomato___Early_blight")
        confidence: Model confidence score
        language: Language code (en, hi, mr)
        force_offline: Skip LLM call, use dictionary only
    
    Returns:
        Remedy dictionary matching REMEDY_SCHEMA
    """
    generator = LLMRemedyGenerator()
    
    if force_offline:
        result = generator.dictionary.get(class_name, language)
        if result:
            return result
        return generator._default_remedy(
            class_name.split("___")[0].replace("_", " "),
            class_name.split("___")[1].replace("_", " ") if "___" in class_name else "Unknown",
            confidence,
            language
        )
    
    return generator.get_remedy(class_name, confidence, language)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test remedy generation")
    parser.add_argument("--class", dest="class_name", type=str, required=True)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--lang", type=str, default="en", choices=["en", "hi", "mr"])
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    
    remedy = get_remedy(args.class_name, args.confidence, args.lang, args.offline)
    print(json.dumps(remedy, indent=2, ensure_ascii=False))