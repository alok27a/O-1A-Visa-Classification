from pydantic import BaseModel
from typing import Dict, List

class AssessmentResult(BaseModel):
    rule_based_matches: Dict[str, List[str]]
    rule_based_rating: str
    llm_validated_matches: Dict[str, List[str]]
    llm_based_rating: str
    combined_rating: str