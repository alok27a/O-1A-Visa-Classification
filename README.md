# O-1A Visa Eligibility Assessor

A FastAPI-based system to evaluate CVs against USCIS O-1A visa criteria using hybrid rule-based + LLM validation.

## Features

- **Hybrid Analysis**: Combines regex patterns + Hugging Face LLM
- **8 USCIS Criteria**: Awards, Membership, Press, etc.
- **PDF/DOCX Support**: Processes common CV formats
- **Three-Tier Rating**: Low/Medium/High eligibility confidence

---

Link for the document for proposed architecture [click here](https://docs.google.com/document/d/1PN4EdJPzslOteTq-C48cHROkTSAEQ5s6AWJsw2LQR7A/edit?usp=sharing)

## Quick Start

### 1. Clone Repo
```bash
git clone https://github.com/alok27a/O-1A-Visa-Classification
cd O-1A-Visa-Classification
```
## 2. Install Dependencies
```bash
pip install -r requirements.txt
```
## 3. Configure Environment

Create `.env` file:
```bash
HF_API_KEY=your_huggingface_api_key
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.2
HF_API_URL=https://api-inference.huggingface.co
```
## 4. Start Server
```bash
uvicorn main:app --reload
```

## Usage

### Test with Sample CV
Can use the FastAPI docs.
```bash
{
  "rule_based_matches": {
    "awards": [],
    "membership": [],
    "press": [],
    "original_contribution": [],
    "critical_employment": []
  },
  "rule_based_rating": "medium",
  "llm_validated_matches": {
    "awards": [],
    "membership": [],
    "press": [],
    "judging": [],
    "original_contribution": [],
    "scholarly_articles": [],
    "critical_employment": [],
    "high_remuneration": []
  },
  "llm_based_rating": "low",
  "combined_rating": "low"
}
```
## Contact Details

### Alok Mathur
[`E-Mail`](mailto:am6499@columbia.edu)
[`LinkedIn`](https://www.linkedin.com/in/alok-mathur-5aab4534/)