import re
import requests
import PyPDF2
import docx
from io import BytesIO
from typing import Dict, List, Any
from config import settings
from schemas import AssessmentResult

# ----------------------
# CV Processing
# ----------------------
async def process_cv(file: UploadFile) -> str:
    """Extract text from PDF/DOCX files"""
    content = await file.read()
    text = ""
    
    try:
        if file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = "".join(page.extract_text() for page in pdf_reader.pages)
        elif file.filename.endswith(('.docx', '.doc')):
            doc = docx.Document(BytesIO(content))
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")
    
    return text

# ----------------------
# Rule-Based Checks
# ----------------------
CRITERIA_PATTERNS = {
    "awards": [
        r'\b(Award|Prize|Honor|Nobel|Grammy|Forbes|Top\s*\d+%)\b',
        r'[A-Z][a-z]+ (Award|Prize)'
    ],
    "membership": [
        r'\b(IEEE|ACM|National Academy|Fellow|Board Member|Chair)\b'
    ],
    # ... patterns for other criteria ...
}

def rule_based_scan(text: str) -> Dict[str, List[str]]:
    """Initial regex-based matching"""
    results = {}
    
    for criterion, patterns in CRITERIA_PATTERNS.items():
        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
        results[criterion] = list(set(matches))
    
    return results

# ----------------------
# LLM Integration
# ----------------------
LLM_PROMPTS = {
    "awards": """Evaluate if this describes a nationally recognized award: {text}""",
    # ... other criteria prompts ...
}

def query_llm(prompt: str) -> str:
    """Query Hugging Face Inference API"""
    try:
        response = requests.post(
            f"{settings.hf_api_url}/models/{settings.hf_model}",
            headers={"Authorization": f"Bearer {settings.hf_api_key}"},
            json={"inputs": prompt}
        )
        return response.json()[0]["generated_text"].lower()
    except Exception as e:
        raise RuntimeError(f"LLM query failed: {str(e)}")

# ----------------------
# Hybrid Evaluation
# ----------------------
async def hybrid_evaluation(text: str) -> AssessmentResult:
    """Main evaluation workflow"""
    # Rule-based matching
    rule_matches = rule_based_scan(text)
    
    # LLM validation
    llm_validated = await validate_with_llm(rule_matches, text)
    
    # Calculate rating
    rating = calculate_rating(llm_validated)
    
    return AssessmentResult(
        rule_based_matches=rule_matches,
        llm_validated_matches=llm_validated,
        rating=rating
    )