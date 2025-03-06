# In config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    hf_api_url: str
    hf_model: str
    hf_api_key: str
    
    class Config:
        env_file = ".env"

settings = Settings()

# Verify configuration on startup
if not all([settings.hf_api_url, settings.hf_model, settings.hf_api_key]):
    raise ValueError("Missing Hugging Face configuration in .env file")