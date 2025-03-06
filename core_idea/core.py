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
