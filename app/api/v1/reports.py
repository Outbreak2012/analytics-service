from fastapi import APIRouter, Depends, HTTPException, Query
from datetime import datetime, timedelta
from app.models.schemas import KPIResponse, ReportRequest, ReportResponse
from app.core.security import get_current_user
import logging
import random
import uuid
from app.core.config import settings
from app.db.redis_cache import redis_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reports", tags=["Reports & KPIs"])


@router.get("/kpis", response_model=KPIResponse)
async def get_kpis(
    current_user: dict = Depends(get_current_user)
):
    """Get current KPIs dashboard"""
    try:
        logger.info("Getting KPIs...")
        cache_key = "kpis_v1"

        # Try to get from cache (fail gracefully if Redis not available)
        try:
            cached = redis_conn.get(cache_key)
            if cached:
                logger.info("✅ KPIs returned from cache")
                return KPIResponse(**cached)
        except Exception:
            logger.warning("⚠️ Redis not available, continuing without cache")

        # Generate mock KPI data
        kpi = KPIResponse(
            total_passengers=random.randint(5000, 10000),
            total_revenue=random.uniform(50000, 100000),
            avg_occupancy=random.uniform(0.6, 0.9),
            routes_active=random.randint(10, 25),
            peak_hour=random.choice([7, 8, 17, 18]),
            sentiment_avg=random.uniform(0.6, 0.85),
            generated_at=datetime.now()
        )

        # Cache result (store ISO string for datetime)
        try:
            payload = kpi.dict()
            if isinstance(payload.get('generated_at'), datetime):
                payload['generated_at'] = payload['generated_at'].isoformat()
            redis_conn.set(cache_key, payload, ttl=getattr(settings, 'CACHE_TTL', 300))
            logger.info("✅ KPIs cached")
        except Exception:
            logger.warning("⚠️ Could not write KPIs to Redis cache")

        return kpi
        
    except Exception as e:
        logger.error(f"Error getting KPIs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
async def get_dashboard(
    period: str = Query("daily", regex="^(daily|weekly|monthly)$"),
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive dashboard data"""
    try:
        logger.info(f"Getting {period} dashboard...")

        cache_key = f"dashboard:{period}"

        try:
            cached = redis_conn.get(cache_key)
            if cached:
                logger.info("✅ Dashboard returned from cache")
                return cached
        except Exception:
            logger.warning("⚠️ Redis not available, continuing without cache")

        # Mock dashboard data
        dashboard = {
            "period": period,
            "metrics": {
                "total_passengers": random.randint(5000, 15000),
                "total_revenue": random.uniform(50000, 150000),
                "total_trips": random.randint(2000, 5000),
                "avg_occupancy": random.uniform(0.65, 0.85),
                "on_time_performance": random.uniform(0.80, 0.95)
            },
            "routes": {
                "total": random.randint(15, 30),
                "active": random.randint(12, 25),
                "peak_routes": [
                    {"route_id": i, "name": f"Ruta {i}", "passengers": random.randint(500, 1500)}
                    for i in range(1, 6)
                ]
            },
            "trends": {
                "passenger_growth": random.uniform(-5, 15),
                "revenue_growth": random.uniform(-3, 20),
                "satisfaction_score": random.uniform(0.70, 0.90)
            },
            "alerts": [
                {
                    "type": "warning",
                    "message": "Ruta 5 con alta demanda",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "type": "info",
                    "message": "Mantenimiento programado Ruta 12",
                    "timestamp": datetime.now().isoformat()
                }
            ],
            "generated_at": datetime.now()
        }
        
        # Try to cache the generated dashboard (do not fail if Redis missing)
        try:
            to_cache = dict(dashboard)
            if isinstance(to_cache.get('generated_at'), datetime):
                to_cache['generated_at'] = to_cache['generated_at'].isoformat()
            redis_conn.set(cache_key, to_cache, ttl=getattr(settings, 'CACHE_TTL', 300))
            logger.info("✅ Dashboard cached")
        except Exception:
            logger.warning("⚠️ Could not write dashboard to Redis cache")

        return dashboard

    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=ReportResponse)
async def generate_report(
    request: ReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate custom report"""
    try:
        logger.info(f"Generating {request.report_type} report...")
        
        report_id = str(uuid.uuid4())
        
        # Simulate report generation
        return ReportResponse(
            report_id=report_id,
            report_type=request.report_type,
            status="completed",
            download_url=f"/api/v1/reports/download/{report_id}",
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{report_id}")
async def download_report(
    report_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Download generated report"""
    try:
        logger.info(f"Downloading report {report_id}...")
        
        # Mock report data
        return {
            "report_id": report_id,
            "status": "ready",
            "data": {
                "summary": "Reporte generado exitosamente",
                "records": random.randint(100, 1000),
                "generated_at": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_performance_metrics(
    route_id: int = Query(None, description="Filter by route"),
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_user)
):
    """Get performance metrics"""
    try:
        logger.info(f"Getting performance metrics for {days} days...")
        
        metrics = {
            "on_time_rate": random.uniform(0.80, 0.95),
            "avg_delay_minutes": random.uniform(2, 8),
            "passenger_satisfaction": random.uniform(0.70, 0.90),
            "revenue_per_trip": random.uniform(50, 150),
            "occupancy_rate": random.uniform(0.60, 0.85),
            "period": {
                "days": days,
                "from": (datetime.now() - timedelta(days=days)).isoformat(),
                "to": datetime.now().isoformat()
            },
            "generated_at": datetime.now()
        }
        
        if route_id:
            metrics["route_id"] = route_id
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/revenue")
async def get_revenue_analysis(
    days: int = Query(7, ge=1, le=90),
    current_user: dict = Depends(get_current_user)
):
    """Get revenue analysis"""
    try:
        logger.info(f"Getting revenue analysis for {days} days...")
        
        # Generate daily revenue data
        daily_revenue = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i - 1)
            daily_revenue.append({
                "date": date.date().isoformat(),
                "revenue": random.uniform(5000, 15000),
                "transactions": random.randint(200, 600),
                "avg_transaction": random.uniform(20, 50)
            })
        
        total_revenue = sum(day["revenue"] for day in daily_revenue)
        avg_daily = total_revenue / days
        
        return {
            "summary": {
                "total_revenue": total_revenue,
                "avg_daily_revenue": avg_daily,
                "total_transactions": sum(day["transactions"] for day in daily_revenue),
                "growth_rate": random.uniform(-5, 15)
            },
            "daily_breakdown": daily_revenue,
            "period": {
                "days": days,
                "from": (datetime.now() - timedelta(days=days)).isoformat(),
                "to": datetime.now().isoformat()
            },
            "generated_at": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting revenue analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
