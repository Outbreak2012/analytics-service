from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import sys
from datetime import datetime
import os
import pickle

from app.core.config import settings
from app.api.v1 import demand, segmentation, sentiment, reports, metrics, testing
from app.db.clickhouse import clickhouse_conn
from app.db.mongodb import mongodb_conn
from app.db.redis_cache import redis_conn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Analytics & Reporting Service para CityTransit con ML/DL",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now().isoformat()
    }


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    logger.info("=" * 50)
    logger.info(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("=" * 50)
    
    try:
        # Connect to databases
        logger.info("📡 Connecting to databases...")
        
        try:
            clickhouse_conn.connect()
        except Exception as e:
            logger.warning(f"⚠️ ClickHouse connection failed: {e}")
        
        try:
            mongodb_conn.connect()
        except Exception as e:
            logger.warning(f"⚠️ MongoDB connection failed: {e}")
        
        try:
            redis_conn.connect()
        except Exception as e:
            logger.warning(f"⚠️ Redis connection failed: {e}")
        
        logger.info("✅ Database connections initialized")
        
        # Load ML models (optional - service works without them)
        logger.info("🤖 Loading ML models (optional)...")
        
        try:
            from app.ml.lstm_model import lstm_predictor
            lstm_predictor.load_or_create_model()
            logger.info("✅ LSTM model loaded")
        except ImportError as e:
            logger.warning(f"⚠️ LSTM module not available: {e}. Install tensorflow to enable.")
        except Exception as e:
            logger.warning(f"⚠️ LSTM model load failed: {e}. Will use fallback predictions.")
        
        try:
            from app.ml.bert_model import bert_analyzer
            bert_analyzer.load_model()
            logger.info("✅ BERT model loaded")
        except ImportError as e:
            logger.warning(f"⚠️ BERT module not available: {e}. Install transformers to enable.")
        except Exception as e:
            logger.warning(f"⚠️ BERT model load failed: {e}. Will use fallback sentiment analysis.")

        # Load DBSCAN model if persisted (optional)
        try:
            dbscan_path = os.path.join(settings.MODEL_PATH, 'dbscan_segmentation.pkl')
            if os.path.exists(dbscan_path):
                from app.ml import dbscan_model as dbscan_module
                with open(dbscan_path, 'rb') as f:
                    loaded = pickle.load(f)
                    dbscan_module.dbscan_segmentation = loaded
                    logger.info(f"✅ DBSCAN model loaded from {dbscan_path}")
            else:
                logger.info("ℹ️  No persisted DBSCAN model found. Will create on demand.")
        except Exception as e:
            logger.warning(f"⚠️ DBSCAN model load failed: {e}. Will use fresh model.")
        
        logger.info("=" * 50)
        logger.info("✅ Service ready!")
        logger.info(f"📚 API Documentation: http://localhost:8000/docs")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        raise


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Shutting down service...")
    
    try:
        clickhouse_conn.disconnect()
        mongodb_conn.disconnect()
        redis_conn.disconnect()
        logger.info("✅ Connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Include routers
app.include_router(demand.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(segmentation.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(sentiment.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(reports.router, prefix="/api/v1", tags=["Reports"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Model Metrics"])
app.include_router(testing.router, prefix="/api/v1", tags=["Testing - Realistic Data"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "api": "/api/v1"
        }
    }


# Test endpoint without auth for frontend connection testing
@app.get("/api/v1/test/connection")
async def test_connection():
    """Test endpoint without authentication"""
    return {
        "status": "connected",
        "service": settings.APP_NAME,
        "message": "Analytics service is reachable from frontend",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
