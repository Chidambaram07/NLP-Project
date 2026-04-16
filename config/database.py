from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from config.settings import settings


class Database:
    """MongoDB connection manager"""
    
    client: AsyncIOMotorClient = None
    sync_client: MongoClient = None
    
    @classmethod
    def connect(cls):
        """Connect to MongoDB (for scraper - synchronous)"""
        cls.sync_client = MongoClient(settings.mongodb_uri)
        print(f"✅ Connected to MongoDB: {settings.mongodb_database}")
        return cls.sync_client[settings.mongodb_database]
    
    @classmethod
    async def connect_async(cls):
        """Connect to MongoDB (for FastAPI - async)"""
        cls.client = AsyncIOMotorClient(settings.mongodb_uri)
        print(f"✅ Connected to MongoDB (Async): {settings.mongodb_database}")
        return cls.client[settings.mongodb_database]
    
    @classmethod
    def close(cls):
        """Close MongoDB connection"""
        if cls.sync_client:
            cls.sync_client.close()
        if cls.client:
            cls.client.close()
        print("🔌 MongoDB connection closed")
    
    @classmethod
    def get_database(cls):
        """Get synchronous database instance"""
        if not cls.sync_client:
            return cls.connect()
        return cls.sync_client[settings.mongodb_database]
    
    @classmethod
    async def get_async_database(cls):
        """Get async database instance"""
        if not cls.client:
            return await cls.connect_async()
        return cls.client[settings.mongodb_database]


# Create database instance
db = Database.get_database()