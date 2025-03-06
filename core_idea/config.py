import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    hf_api_key: str = os.getenv("HF_API_KEY")
    hf_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    hf_api_url: str = "https://api-inference.huggingface.co"
    
    class Config:
        env_file = ".env"

settings = Settings()