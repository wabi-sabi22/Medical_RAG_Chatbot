# src/config.py
import os
from dotenv import load_dotenv

# Tải các biến môi trường từ file .env vào hệ thống
load_dotenv()

class Settings:
    # --- API KEYS ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
   # OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    HF_TOKEN: str = os.getenv("HF_TOKEN")
   #GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    # --- QDRANT DATABASE ---
    QDRANT_URL: str = os.getenv("QDRANT_URL")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY")
   
    # --- AI MODELS CONFIG ---
    # Đồng bộ hóa mặc định.
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "embed-multilingual-v3.0")
    #EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    # Đồng bộ hóa mặc định 
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")


    LLAMA_CLOUD_API_KEY: str = os.getenv("LLAMA_CLOUD_API_KEY")

    MEMGRAPH_URL: str = os.getenv("MEMGRAPH_URL", "bolt://memgraph:7687")
# Khởi tạo object settings
settings = Settings()