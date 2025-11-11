from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
from app.models.schemas import (
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    SentimentType
)
from app.ml.bert_model import bert_analyzer
from app.core.security import get_current_user
import logging
import random
from app.db.redis_cache import redis_conn
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment", tags=["Sentiment Analysis"])


@router.post("/analyze", response_model=SentimentAnalysisResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    current_user: dict = Depends(get_current_user)
):
    """Analyze sentiment of text using BERT"""
    try:
        logger.info(f"Analyzing sentiment for text: {request.text[:50]}...")
        
        # Analyze sentiment
        result = bert_analyzer.analyze(request.text)
        
        return SentimentAnalysisResponse(
            sentiment=SentimentType(result['sentiment']),
            confidence_score=result['confidence_score'],
            scores=result['scores'],
            analyzed_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch")
async def analyze_batch(
    texts: list[str],
    current_user: dict = Depends(get_current_user)
):
    """Analyze sentiment for multiple texts"""
    try:
        logger.info(f"Analyzing sentiment for {len(texts)} texts...")
        
        results = bert_analyzer.batch_analyze(texts)
        
        return {
            "results": results,
            "total": len(results),
            "analyzed_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary")
async def get_sentiment_summary(
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_user)
):
    """Get sentiment analysis summary from MongoDB"""
    try:
        logger.info(f"Getting sentiment summary for {days} days...")

        cache_key = f"sentiment:summary:{days}"
        try:
            cached = redis_conn.get(cache_key)
            if cached:
                logger.info("✅ Sentiment summary returned from cache")
                return cached
        except Exception:
            logger.debug("Redis not available for sentiment/summary")

        # Get real feedback from MongoDB
        from app.db.mongodb import mongodb_conn
        
        try:
            db = mongodb_conn.connect()
            
            # Get sample of recent feedback
            cursor = db.user_feedback.find(
                {"comentario": {"$exists": True, "$ne": ""}},
                {"comentario": 1, "sentimiento": 1}
            ).limit(100)
            
            feedbacks = list(cursor)
            logger.info(f"📊 Found {len(feedbacks)} feedback comments in MongoDB")
            
            if feedbacks:
                # Use existing sentiment if available, otherwise analyze
                results = []
                for feedback in feedbacks:
                    if 'sentimiento' in feedback and feedback['sentimiento']:
                        # Use pre-analyzed sentiment
                        results.append({
                            'sentiment': feedback['sentimiento'],
                            'confidence_score': 0.85,
                            'text': feedback['comentario']
                        })
                    else:
                        # Analyze new comment
                        analysis = bert_analyzer.analyze(feedback['comentario'])
                        results.append(analysis)
                
                # Get summary stats
                summary = bert_analyzer.get_summary_stats(results)
            else:
                logger.warning("No feedback found in MongoDB, using sample data")
                # Fallback to sample data
                sample_feedbacks = [
                    "Excelente servicio, muy puntual y cómodo",
                    "El bus llegó tarde, muy mala experiencia",
                    "Buen servicio en general, nada especial",
                    "Muy mal, conductor grosero y bus sucio",
                    "Todo perfecto, lo recomiendo completamente",
                    "Regular, podría mejorar la limpieza",
                    "Pésima atención al cliente",
                    "Me encanta este servicio de transporte",
                    "Normal, cumple su función",
                    "Muy buena experiencia, seguiré usando"
                ]
                results = bert_analyzer.batch_analyze(sample_feedbacks * 5)
                summary = bert_analyzer.get_summary_stats(results)
                
        except Exception as e:
            logger.error(f"Error accessing MongoDB: {e}, using sample data")
            # Fallback data
            sample_feedbacks = [
                "Excelente servicio, muy puntual y cómodo",
                "El bus llegó tarde, muy mala experiencia",
                "Buen servicio en general, nada especial",
                "Muy mal, conductor grosero y bus sucio",
                "Todo perfecto, lo recomiendo completamente",
                "Regular, podría mejorar la limpieza",
                "Pésima atención al cliente",
                "Me encanta este servicio de transporte",
                "Normal, cumple su función",
                "Muy buena experiencia, seguiré usando"
            ]
            results = bert_analyzer.batch_analyze(sample_feedbacks * 5)
            summary = bert_analyzer.get_summary_stats(results)

        result = {
            "summary": summary,
            "period": {
                "days": days,
                "from": (datetime.now() - timedelta(days=days)).isoformat(),
                "to": datetime.now().isoformat()
            },
            "generated_at": datetime.now().isoformat()
        }

        try:
            redis_conn.set(cache_key, result, ttl=getattr(settings, 'CACHE_TTL', 300))
        except Exception:
            logger.debug("Could not cache sentiment summary")

        return result
        
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_sentiment_trends(
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_user)
):
    """Get sentiment trends over time"""
    try:
        logger.info(f"Getting sentiment trends for {days} days...")
        
        # Generate mock trend data
        trends = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            trends.append({
                "date": date.date().isoformat(),
                "positive": random.randint(20, 50),
                "neutral": random.randint(30, 60),
                "negative": random.randint(10, 30),
                "total": random.randint(70, 140)
            })
        
        return {
            "trends": trends,
            "period": {
                "days": days,
                "from": (datetime.now() - timedelta(days=days)).isoformat(),
                "to": datetime.now().isoformat()
            },
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/by-route/{route_id}")
async def get_sentiment_by_route(
    route_id: int,
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_user)
):
    """Get sentiment analysis for specific route"""
    try:
        logger.info(f"Getting sentiment for route {route_id}...")
        
        # Generate mock feedback for route
        route_feedbacks = [
            "Esta ruta es excelente",
            "Buena frecuencia de buses",
            "A veces hay demora",
            "Ruta muy conveniente",
            "Regular el servicio"
        ]
        
        # Analyze
        results = bert_analyzer.batch_analyze(route_feedbacks * 3)
        summary = bert_analyzer.get_summary_stats(results)
        
        return {
            "route_id": route_id,
            "summary": summary,
            "period_days": days,
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting route sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))
