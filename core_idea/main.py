from fastapi import FastAPI, File, UploadFile, HTTPException  # Add HTTPException here
from fastapi.responses import JSONResponse
from core import process_cv, hybrid_evaluation
from schemas import AssessmentResult
import config


app = FastAPI()

@app.post("/assess", response_model=AssessmentResult)
async def assess_o1a(cv: UploadFile = File(...)):
    """Endpoint for O-1A visa assessment"""
    try:
        # Process uploaded CV
        text = await process_cv(cv)
        
        # Perform hybrid evaluation
        result = await hybrid_evaluation(text)
        
        return result
        
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Assessment failed: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)