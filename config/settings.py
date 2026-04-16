from dotenv import load_dotenv
load_dotenv()

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Reddit API
    reddit_client_id: str
    reddit_client_secret: str
    reddit_user_agent: str

    # MongoDB
    mongodb_uri: str
    mongodb_database: str = "reddit_pain_points"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_title: str = "Reddit Pain Points API"
    api_version: str = "1.0.0"

    # Scraping
    max_posts_per_subreddit: int = 200
    scrape_time_filter: str = "month"

    class Config:
        env_file = ".env"
        case_sensitive = False
        # ALLOW extra fields that are not in .env
        extra = "allow"

# Create a settings instance
settings = Settings()
