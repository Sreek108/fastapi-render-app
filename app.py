"""
Lead Intelligence API - Complete Platform
Combines ML Lead Scoring + Geographical Analysis
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, Any
import logging
from datetime import datetime

from ml_engine import AIMLModelsEngine
from geo_engine import GeographicalAnalysisEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Settings
class Settings(BaseSettings):
    DATABASE_SERVER: str = Field(..., env='DATABASE_SERVER')
    DATABASE_NAME: str = Field(..., env='DATABASE_NAME')
    DATABASE_USERNAME: str = Field(..., env='DATABASE_USERNAME')
    DATABASE_PASSWORD: str = Field(..., env='DATABASE_PASSWORD')
    API_TITLE: str = Field(default="Lead Intelligence API", env='API_TITLE')
    API_VERSION: str = Field(default="2.0.0", env='API_VERSION')
    
    class Config:
        env_file = ".env"


settings = Settings()

# FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="🎯 **Complete Lead Intelligence Platform** - ML Lead Scoring, Churn Prediction, Segmentation, Recommendations & Geographical Market Analysis",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ROOT & HEALTH ENDPOINTS
# ============================================================================

@app.get("/", tags=["Root"])
def root():
    """
    🏠 **Root Endpoint** - API Information
    """
    return {
        "service": "Lead Intelligence API",
        "version": settings.API_VERSION,
        "status": "active",
        "description": "Complete Lead Intelligence Platform with ML Models & Geographical Analysis",
        "documentation": "/docs",
        "features": {
            "ml_models": [
                "Lead Scoring",
                "Churn Risk Prediction",
                "Lead Segmentation",
                "Smart Recommendations"
            ],
            "geographical_analysis": [
                "Country Performance",
                "Regional Analysis",
                "Market Recommendations",
                "Market Concentration"
            ]
        },
        "endpoints": {
            "ml_complete": "POST /api/v1/score-all-leads",
            "ml_summary": "GET /api/v1/summary",
            "ml_top_leads": "GET /api/v1/top-leads/{limit}",
            "ml_at_risk": "GET /api/v1/at-risk-leads",
            "ml_recommendations": "GET /api/v1/recommendations",
            "geo_complete": "POST /api/v1/geographical-analysis",
            "geo_countries": "GET /api/v1/countries",
            "geo_recommendations": "GET /api/v1/market-recommendations",
            "health": "GET /health"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """
    ❤️ **Health Check** - Verify API and database connectivity
    """
    try:
        ml_engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        # Test ML database connection
        ml_status = ml_engine.connect_db()
        
        # Test Geo database connection
        geo_engine = GeographicalAnalysisEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        geo_status = geo_engine.connect()
        
        db_status = "connected" if (ml_status and geo_status) else "disconnected"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": db_status,
            "ml_engine": "operational" if ml_status else "unavailable",
            "geo_engine": "operational" if geo_status else "unavailable"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": "disconnected",
            "error": str(e)
        }


# ============================================================================
# ML MODELS ENDPOINTS
# ============================================================================

@app.post("/api/v1/score-all-leads", tags=["ML Models"])
def score_all_leads():
    """
    🎯 **Complete ML Analysis** - Score all active leads
    
    Runs all 4 ML models:
    - Lead Scoring (0-100)
    - Churn Risk Prediction
    - Lead Segmentation
    - Smart Recommendations
    
    **Response Time**: ~10-15 seconds for 1000+ leads
    
    **Returns**: Complete analysis with scores, segments, and recommendations
    """
    try:
        logger.info("Running complete ML analysis on all leads...")
        
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error', 'ML analysis failed')
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/summary", tags=["ML Models"])
def get_summary():
    """
    📊 **Quick Summary** - Get high-level metrics
    
    **Response Time**: ~10 seconds
    
    **Returns**: Total leads, avg score, priority distribution, at-risk count
    """
    try:
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        
        return {
            "summary": results.get('summary', {}),
            "timestamp": results.get('timestamp'),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/top-leads/{limit}", tags=["ML Models"])
def get_top_leads(limit: int = 10):
    """
    🏆 **Top Leads** - Get highest scoring leads
    
    **Parameters**:
    - limit: Number of leads to return (default: 10, max: 100)
    
    **Response Time**: ~10 seconds
    """
    if limit > 100:
        raise HTTPException(status_code=400, detail="Limit cannot exceed 100")
    
    try:
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        top_leads = results.get('top_leads', [])[:limit]
        
        return {
            "top_leads": top_leads,
            "count": len(top_leads),
            "timestamp": results.get('timestamp'),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Top leads retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/at-risk-leads", tags=["ML Models"])
def get_at_risk_leads():
    """
    ⚠️ **At-Risk Leads** - Get leads with high churn risk
    
    **Response Time**: ~10 seconds
    
    **Returns**: Leads that need immediate attention
    """
    try:
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        
        return {
            "at_risk_leads": results.get('at_risk_leads', []),
            "count": len(results.get('at_risk_leads', [])),
            "timestamp": results.get('timestamp'),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"At-risk leads retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/recommendations", tags=["ML Models"])
def get_recommendations():
    """
    💡 **Smart Recommendations** - Get AI-powered action items
    
    **Response Time**: ~10 seconds
    
    **Returns**: Prioritized recommendations for sales team
    """
    try:
        engine = AIMLModelsEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = engine.run_all_models()
        
        return {
            "recommendations": results.get('recommendations', []),
            "count": len(results.get('recommendations', [])),
            "timestamp": results.get('timestamp'),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Recommendations generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# GEOGRAPHICAL ANALYSIS ENDPOINTS
# ============================================================================

@app.post("/api/v1/geographical-analysis", tags=["Geographical Analysis"])
def complete_geographical_analysis():
    """
    🌍 **Complete Geographical Analysis** - Market intelligence & performance
    
    Analyzes:
    - Lead distribution by country
    - Regional performance metrics
    - Market concentration
    - Country-specific recommendations
    
    **Response Time**: ~5-10 seconds
    
    **Returns**: Complete market analysis with countries, regions, and insights
    """
    try:
        logger.info("Running geographical analysis...")
        
        geo_engine = GeographicalAnalysisEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = geo_engine.run_complete_analysis()
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error', 'Geographical analysis failed')
            )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Geographical analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/countries", tags=["Geographical Analysis"])
def get_country_analysis():
    """
    🌍 **Country Analysis** - Get country-level metrics only
    
    **Response Time**: ~5 seconds
    
    **Returns**: Country performance metrics and rankings
    """
    try:
        geo_engine = GeographicalAnalysisEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = geo_engine.run_complete_analysis()
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error')
            )
        
        return {
            "country_analysis": results.get('country_analysis', {}),
            "timestamp": results['timestamp'],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Country analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.get("/api/v1/market-recommendations", tags=["Geographical Analysis"])
def get_market_recommendations():
    """
    💡 **Market Recommendations** - Get geographical insights
    
    **Response Time**: ~5 seconds
    
    **Returns**: Market-specific strategic recommendations
    """
    try:
        geo_engine = GeographicalAnalysisEngine(
            server=settings.DATABASE_SERVER,
            database=settings.DATABASE_NAME,
            username=settings.DATABASE_USERNAME,
            password=settings.DATABASE_PASSWORD
        )
        
        results = geo_engine.run_complete_analysis()
        
        if results.get('status') == 'failed':
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results.get('error')
            )
        
        return {
            "recommendations": results.get('recommendations', []),
            "summary": results.get('summary', {}),
            "timestamp": results['timestamp'],
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Market recommendations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
