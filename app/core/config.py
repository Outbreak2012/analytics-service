from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "CityTransit Analytics Service"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # ClickHouse
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 8123
    CLICKHOUSE_USER: str = "default"
    CLICKHOUSE_PASSWORD: str = "redfire007"
    CLICKHOUSE_DATABASE: str = "paytransit"
    
    # MongoDB
    MONGODB_HOST: str = "localhost"
    MONGODB_PORT: int = 27017
    MONGODB_USER: str = "admin"
    MONGODB_PASSWORD: str = "redfire007"
    MONGODB_DATABASE: str = "paytransit"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = "redfire007"
    REDIS_DB: int = 0
    
    # Java Backend
    BACKEND_URL: str = "http://localhost:8080"
    BACKEND_API_KEY: Optional[str] = None
    
    # JWT
    JWT_SECRET: str = "your-secret-key-minimum-256-bits"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # ML Models
    MODEL_PATH: str = "./models"
    LSTM_MODEL_PATH: str = "./models/lstm_demand_prediction.h5"
    BERT_MODEL_PATH: str = "./models/bert_sentiment_analysis"
    
    # Cache
    CACHE_TTL: int = 3600
    
    # CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:4200",
        "http://localhost:8080",
    ]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
