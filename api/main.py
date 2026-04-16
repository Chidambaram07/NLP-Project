from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from api.routes import router


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="API for Reddit pain points and opportunity detection"
)

# CORS middleware (allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Reddit Pain Points API",
        "version": settings.api_version,
        "endpoints": {
            "docs": "/docs",
            "posts": "/api/v1/posts",
            "pain_points": "/api/v1/pain-points",
            "statistics": "/api/v1/statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    from config.database import db
    
    try:
        # Test database connection
        db.command("ping")
        db_status = "connected"
    except:
        db_status = "disconnected"
    
    return {
        "status": "ok",
        "database": db_status,
        "timestamp": "2026-01-26T12:00:00Z"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )