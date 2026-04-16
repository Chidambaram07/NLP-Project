from pymongo import MongoClient
from config.settings import settings

uri = settings.mongodb_uri

if not uri:
    raise RuntimeError("❌ MONGODB_URI is not loaded. Check .env location.")

print("✅ MongoDB URI loaded")

try:
    client = MongoClient(uri)
    db = client[settings.mongodb_database]
    print("✅ MongoDB connected successfully!")
    print("Collections:", db.list_collection_names())
except Exception as e:
    print("❌ MongoDB connection failed:", e)
