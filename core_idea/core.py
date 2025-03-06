from fastapi import UploadFile, HTTPException
from typing import Dict, List, Any
import re
import requests
import PyPDF2
import docx
from io import BytesIO
from config import settings
from schemas import AssessmentResult

# ----------------------
# Enhanced CV Processing
# ----------------------
async def process_cv(file: UploadFile) -> str:
    """Improved text extraction with error handling"""
    try:
        content = await file.read()
        text = ""
        
        if file.filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(BytesIO(content))
            text = "".join(page.extract_text() for page in pdf_reader.pages)
        elif file.filename.endswith(('.docx', '.doc')):
            doc = docx.Document(BytesIO(content))
            text = "\n".join(para.text for para in doc.paragraphs)
        else:
            raise ValueError("Unsupported file format")
            
        return text.lower()  # Normalize text for matching
        
    except PyPDF2.errors.PdfReadError:
        raise HTTPException(400, "Invalid PDF file")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

# ----------------------
# Enhanced Rule-Based Checks
# ----------------------
CRITERIA_PATTERNS = {
    "awards": [
        r'\b(award|prize|honor|scholarship|competition winner|top[\s-]*\d+%|excellence)\b',
        r'(?i)\b(nobel|grammy|emmy|oscar|pulitzer|forbes\s30|hackathon\swinner)\b'
    ],
    "membership": [
        r'\b(member of|fellowship|invited member|board of|selection committee)\b',
        r'(?i)\b(IEEE|ACM|National Academy|Royal Society|<\s*5% acceptance)\b'
    ],
    "critical_employment": [
        r'\b(lead|principal|chief|director|head of|key (role|position)|senior\s\w+)\b',
        r'(?i)\b(United Nations|CERN|NASA|MIT|Google|Fortune 500|Nobel laureate team)\b'
    ]
}


def rule_based_scan(text: str) -> Dict[str, List[str]]:
    """Improved matching with context awareness"""
    results = {}
    
    for criterion, patterns in CRITERIA_PATTERNS.items():
        matches = []
        for pattern in patterns:
            # Find full line containing matches
            for match in re.finditer(pattern, text, re.IGNORECASE):
                start = text.rfind('\n', 0, match.start()) + 1
                end = text.find('\n', match.end())
                full_line = text[start:end].strip()
                if full_line not in matches:
                    matches.append(full_line)
        results[criterion] = matches
    
    return results

# ----------------------
# Enhanced LLM Integration
# ----------------------
LLM_PROMPTS = {
    "awards": """Does this describe a nationally/internationally recognized prize or award requiring exceptional achievement in the field? 
    Consider: Competition wins, prestigious scholarships, industry-specific honors. 
    Text: {text} 
    Answer (yes/no):""",

    "membership": """Does this indicate membership in an association that:
    1. Requires outstanding achievements for admission?
    2. Has selection criteria judged by field experts?
    3. Maintains <5% acceptance rate?
    Text: {text} 
    Answer (yes/no):""",

    "press": """Is this coverage in professional/trade publications or major media discussing work achievements?
    Valid examples: TechCrunch feature, IEEE journal mention, patent citation
    Text: {text} 
    Answer (yes/no):""",

    "judging": """Does this show evaluation of others' work through:
    1. Competition/job/research judging?
    2. Peer review for distinguished venues?
    3. Editorial board membership?
    Text: {text} 
    Answer (yes/no):""",

    "original_contribution": """Does this describe an original contribution of major significance through:
    1. Patented inventions?
    2. Field-changing techniques?
    3. Widely adopted systems?
    Text: {text} 
    Answer (yes/no):""",

    "scholarly_articles": """Is this authorship in distinguished professional/trade publications showing field expertise?
    Consider: Journal papers, conference proceedings, invited book chapters
    Text: {text} 
    Answer (yes/no):""",

    "critical_employment": """Does this demonstrate essential roles at organizations with:
    1. National/international reputation?
    2. Field leadership position?
    Examples: Lead at Fortune 500, key researcher at CERN
    Text: {text} 
    Answer (yes/no):""",

    "high_remuneration": """Does this show compensation substantially exceeding field norms through:
    1. Salary/equity documentation?
    2. Top percentile earnings?
    3. Exceptional benefits?
    Text: {text} 
    Answer (yes/no):"""
}


def query_llm(prompt: str) -> str:
    """Improved LLM query with error handling"""
    try:
        response = requests.post(
            f"{settings.hf_api_url}/models/{settings.hf_model}",
            headers={"Authorization": f"Bearer {settings.hf_api_key}"},
            json={"inputs": prompt},
            timeout=10
        )
        if response.status_code != 200:
            return "error"
        return response.json()[0]["generated_text"].lower().strip()
    except Exception as e:
        return "error"

async def validate_with_llm(rule_matches: Dict[str, List[str]], text: str) -> Dict[str, List[str]]:
    """Improved validation with confidence scoring"""
    llm_validated = {c: [] for c in CRITERIA_PATTERNS.keys()}
    
    for criterion, matches in rule_matches.items():
        validated = []
        for match in matches:
            prompt = LLM_PROMPTS.get(criterion, "").format(text=match)
            if not prompt:
                continue
                
            response = query_llm(prompt)
            if "yes" in response:
                validated.append(match)
            elif "error" in response:
                validated.append(f"{match} (validation failed)")
                
        llm_validated[criterion] = validated
    
    return llm_validated

# ----------------------
# Rating Calculations
# ----------------------
def calculate_rule_rating(rule_matches: Dict[str, List[str]]) -> str:
    """Rule-based rating calculation"""
    counts = sum(len(v) > 0 for v in rule_matches.values())
    if counts >= 3: return "medium"
    if counts >= 5: return "high"
    return "low"

def calculate_llm_rating(llm_matches: Dict[str, List[str]]) -> str:
    """USCIS-aligned scoring:
    - Awards/Press/Original Contributions = 3 pts
    - Scholarly/Judging/Critical Employment = 2 pts
    - Membership/Remuneration = 1 pt
    """
    scores = {
        "awards": 3,
        "press": 3,
        "original_contribution": 3,
        "scholarly_articles": 2,
        "judging": 2,
        "critical_employment": 2,
        "membership": 1,
        "high_remuneration": 1
    }
    
    total = sum(
        scores[c] * len(matches)
        for c, matches in llm_matches.items()
        if not any('(validation failed)' in m for m in matches)
    )
    
    if total >= 8: return "high"  # Strong evidence
    if total >= 5: return "medium"  # Probable qualification
    return "low"

# ----------------------
# Enhanced Hybrid Evaluation
# ----------------------
async def hybrid_evaluation(text: str) -> AssessmentResult:
    """Improved evaluation with separate ratings"""
    # Rule-based analysis
    rule_matches = rule_based_scan(text)
    rule_rating = calculate_rule_rating(rule_matches)
    
    # LLM validation
    llm_validated = await validate_with_llm(rule_matches, text)
    llm_rating = calculate_llm_rating(llm_validated)
    
    return AssessmentResult(
        rule_based_matches=rule_matches,
        rule_based_rating=rule_rating,
        llm_validated_matches=llm_validated,
        llm_based_rating=llm_rating,
        combined_rating="high" if "high" in {rule_rating, llm_rating} else "medium"
    )