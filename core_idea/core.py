from fastapi import UploadFile, HTTPException
from typing import Dict, List, Tuple
import re
import requests
import PyPDF2
import docx
from io import BytesIO
from core_idea.config import settings
from core_idea.schemas import AssessmentResult

# ----------------------
# LLM Prompts (USCIS-aligned)
# ----------------------
LLM_PROMPTS = {
    "awards": """Evaluate if the text describes a nationally/internationally recognized achievement:
1. Major competition wins (hackathons, industry challenges)
2. Prestigious scholarships/fellowships
3. Selective recognition lists (Forbes 30, 40 Under 40)
Text: {text}
Answer (yes/no):""",

    "membership": """Does this indicate membership requiring exceptional achievement?
1. <5% acceptance rate organizations
2. Expert-vetted selection processes
3. Leadership in field-specific consortia
Text: {text}
Answer (yes/no):""",

    "press": """Is this coverage in reputable media discussing professional work?
1. Industry-specific publications
2. Major news features/interviews
3. Citations in technical documentation
Text: {text}
Answer (yes/no):""",

    "judging": """Does this show evaluation of others' work?
1. Competition/research judging
2. Peer review activities
3. Grant/patent evaluation
Text: {text}
Answer (yes/no):""",

    "original_contribution": """Does this describe field-impacting innovations?
1. Patented inventions
2. Widely adopted methodologies
3. Novel technical systems
Text: {text}
Answer (yes/no):""",

    "scholarly_articles": """Evidence of authoritative publications?
1. Peer-reviewed journals
2. Conference proceedings
3. Technical white papers
Text: {text}
Answer (yes/no):""",

    "critical_employment": """Key role in distinguished organization?
1. Leadership in Fortune 500/unicorns
2. Essential roles at renowned institutions
3. Strategic impact documentation
Text: {text}
Answer (yes/no):""",

    "high_remuneration": """Compensation significantly above norms?
1. Top percentile earnings
2. Exceptional equity/benefits
3. Industry comparison evidence
Text: {text}
Answer (yes/no):"""
}

# ----------------------
# Enhanced Rule Patterns
# ----------------------
CRITERIA_PATTERNS = {
    "awards": [
        r'\b(award|prize|honor|scholarship|fellowship|top[\s-]*\d+%|finalist|selected|featured)\b',
        r'(?i)\b(hackathon|competition|grants?|recognized as|accolade|distinction)\b'
    ],
    "membership": [
        r'\b(member of|fellowship|invited member|board|advisory|consortium|coalition)\b',
        r'(?i)\b(peer review|selection committee|exclusive|accelerator|<\s*5%)\b'
    ],
    "press": [
        r'\b(featured|interview|article|covered|quoted|mentioned|citation)\b',
        r'(?i)\b(medium|substack|techcrunch|forbes|wired|ted talk|panelist)\b'
    ],
    "original_contribution": [
        r'\b(patent|innovati(on|ve)|breakthrough|pioneer|developed|created|built)\b',
        r'(?i)\b(adopted by|implemented at|integrated into|deployed across)\b'
    ],
    "critical_employment": [
        r'\b(lead|principal|chief|director|architect|founder|strategic|key (role|position))\b',
        r'(?i)\b(unicorn|fortune 500|y combinator|top-tier|mission-critical)\b'
    ]
}


import logging

def query_llm(prompt: str) -> str:
    """Execute LLM query with detailed logging"""
    logger = logging.getLogger(__name__)
    
    try:
        # Log request details
        logger.info(f"Starting LLM request to {settings.hf_api_url}")
        logger.debug(f"Full prompt: {prompt[:200]}...")  # First 200 chars
        
        response = requests.post(
            f"{settings.hf_api_url}/models/{settings.hf_model}",
            headers={"Authorization": f"Bearer {settings.hf_api_key}"},
            json={"inputs": prompt},
            timeout=15
        )
        
        # Log response metadata
        logger.info(f"API response: {response.status_code}")
        logger.debug(f"Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            logger.error(f"API error: {response.text[:500]}...")
            return "error: api_failure"
            
        result = response.json()[0]["generated_text"].lower().strip()
        logger.debug(f"API success: {result[:200]}...")  # Truncate long responses
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error: {str(e)}", exc_info=True)
        return "error: network_failure"
    except KeyError:
        logger.error("Invalid response format - missing 'generated_text'", exc_info=True)
        return "error: invalid_response"
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return f"error: {str(e)}"

# ----------------------
# Core Processing Functions
# ----------------------
async def process_cv(file: UploadFile) -> str:
    """Robust text extraction with format handling"""
    try:
        content = await file.read()
        if file.filename.endswith('.pdf'):
            return extract_pdf_text(content)
        elif file.filename.endswith(('.docx', '.doc')):
            return extract_docx_text(content)
        raise ValueError("Unsupported format")
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

def extract_pdf_text(content: bytes) -> str:
    """PDF text extraction with error recovery"""
    try:
        reader = PyPDF2.PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() for page in reader.pages)
    except PyPDF2.errors.PdfReadError:
        raise HTTPException(400, "Invalid PDF structure")

def extract_docx_text(content: bytes) -> str:
    """DOCX text extraction"""
    doc = docx.Document(BytesIO(content))
    return "\n".join(para.text for para in doc.paragraphs)

# ----------------------
# Context-Aware Analysis
# ----------------------
def rule_based_scan(text: str) -> Dict[str, List[str]]:
    """Pattern matching with context capture"""
    results = {}
    text_lower = text.lower()
    
    for criterion, patterns in CRITERIA_PATTERNS.items():
        matches = []
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                context = get_context_window(text, match.start(), 200)
                if context not in matches:
                    matches.append(context)
        results[criterion] = matches
    return results

# ----------------------
# Section Extraction Helpers
# ----------------------
def extract_section(text: str, headers: List[str]) -> str:
    """Extract resume section content by header"""
    text_lower = text.lower()
    for header in headers:
        start = text_lower.find(header.lower())
        if start != -1:
            end = text_lower.find('\n\n', start)
            return text[start:end].strip() if end != -1 else text[start:].strip()
    return ""

def get_context_window(text: str, position: int, window_size: int = 300) -> str:
    """Get text around a specific position"""
    start = max(0, position - window_size)
    end = min(len(text), position + window_size)
    return text[start:end].strip()

# ----------------------
# Updated Validation Logic
# ----------------------
async def validate_with_llm(rule_matches: Dict[str, List[str]], full_text: str) -> Dict[str, List[str]]:
    """Enhanced validation with section analysis"""
    llm_validated = {c: [] for c in LLM_PROMPTS.keys()}
    section_context = {
        'awards': extract_section(full_text, ['awards', 'honors', 'recognition']),
        'membership': extract_section(full_text, ['membership', 'leadership', 'advisory']),
        'press': extract_section(full_text, ['media', 'publications', 'press']),
        'original_contribution': extract_section(full_text, ['innovations', 'contributions', 'patents'])
    }

    for criterion in LLM_PROMPTS.keys():
        # Validate rule-based matches first
        validated_matches = []
        for match in rule_matches.get(criterion, []):
            if validate_match(criterion, match):
                validated_matches.append(match)
        
        # Section-based validation for missed context
        if not validated_matches:
            context = section_context.get(criterion, full_text[:1000])
            if context and validate_match(criterion, context):
                validated_matches.append(f"Section: {context[:250]}...")
        
        llm_validated[criterion] = validated_matches
    
    return llm_validated

def validate_match(criterion: str, text: str) -> bool:
    """Execute LLM validation for a match"""
    prompt = LLM_PROMPTS[criterion].format(text=text)
    response = query_llm(prompt)
    return "yes" in response.lower()


def analyze_cv_structure(text: str) -> Dict[str, str]:
    """Identify CV sections"""
    sections = {}
    common_headers = ["experience", "education", "awards", "projects", "skills"]
    
    for header in common_headers:
        start = text.lower().find(header)
        if start != -1:
            end = text.find("\n\n", start)
            sections[header] = text[start:end] if end != -1 else text[start:]
    
    return sections

def get_relevant_context(criterion: str, sections: Dict[str, str], fallback: str) -> str:
    """Get criterion-specific context"""
    context_map = {
        "awards": ["awards", "experience"],
        "press": ["experience", "projects"],
        "original_contribution": ["experience", "projects"],
        "critical_employment": ["experience"]
    }
    return " ".join(sections.get(k, "") for k in context_map.get(criterion, [])) or fallback[:1000]


# ----------------------
# USCIS-Aligned Scoring
# ----------------------
def calculate_ratings(rule_matches: Dict[str, List[str]], llm_matches: Dict[str, List[str]]) -> Tuple[str, str, str]:
    """Comprehensive rating calculation"""
    rule_score = sum(len(v) > 0 for v in rule_matches.values())
    llm_score = calculate_llm_score(llm_matches)
    
    return (
        "high" if rule_score >= 4 else "medium" if rule_score >= 2 else "low",
        "high" if llm_score >= 7 else "medium" if llm_score >= 4 else "low",
        "high" if llm_score >= 5 else "medium" if llm_score >= 3 else "low"
    )

def calculate_llm_score(matches: Dict[str, List[str]]) -> int:
    """Weighted USCIS scoring"""
    weights = {
        "awards": 3,
        "original_contribution": 3,
        "critical_employment": 2,
        "press": 2,
        "judging": 2,
        "scholarly_articles": 1,
        "membership": 1,
        "high_remuneration": 1
    }
    return sum(weights[c] * len(m) for c, m in matches.items())

# ----------------------
# API Endpoint
# ----------------------
async def hybrid_evaluation(text: str) -> AssessmentResult:
    """Full evaluation pipeline"""
    rule_matches = rule_based_scan(text)
    llm_validated = await validate_with_llm(rule_matches, text)
    rule_rating, llm_rating, combined = calculate_ratings(rule_matches, llm_validated)
    
    return AssessmentResult(
        rule_based_matches=rule_matches,
        rule_based_rating=rule_rating,
        llm_validated_matches=llm_validated,
        llm_based_rating=llm_rating,
        combined_rating=combined
    )