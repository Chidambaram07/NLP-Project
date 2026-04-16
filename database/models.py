from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class RedditPost(BaseModel):
    """Model for Reddit posts"""
    post_id: str
    subreddit: str
    title: str
    content: str
    author: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    url: str
    is_pain_point: bool
    is_opportunity: bool
    category: Optional[str] = None
    scraped_at: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "post_id": "abc123",
                "subreddit": "developersIndia",
                "title": "Frustrated with job search as a fresher",
                "content": "I've applied to 200+ companies but no response...",
                "author": "user123",
                "score": 45,
                "upvote_ratio": 0.92,
                "num_comments": 23,
                "created_utc": "2026-01-20T10:30:00",
                "url": "https://reddit.com/r/developersIndia/...",
                "is_pain_point": True,
                "is_opportunity": True,
                "category": "Career",
                "scraped_at": "2026-01-26T12:00:00"
            }
        }


class PainPoint(BaseModel):
    """Aggregated pain point model"""
    pain_point_id: Optional[str] = None
    category: str
    description: str
    frequency: int = 1
    related_posts: list = []
    top_keywords: list = []
    opportunity_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "Career",
                "description": "Difficulty finding jobs as a fresher in tech",
                "frequency": 45,
                "related_posts": ["abc123", "def456"],
                "top_keywords": ["job", "fresher", "no response", "interview"],
                "opportunity_score": 8.5
            }
        }


class OpportunitySummary(BaseModel):
    """Summary of detected opportunities"""
    opportunity_id: Optional[str] = None
    title: str
    description: str
    category: str
    pain_points_count: int
    potential_users: int
    average_score: float
    top_subreddits: list
    action_items: list
    created_at: datetime = Field(default_factory=datetime.utcnow)