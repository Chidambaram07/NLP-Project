"""
create_indexes.py
─────────────────
Run ONCE to create MongoDB indexes for:
  1. Text search index on title + content (fixes the search feature)
  2. Unique index on post_id (prevents duplicate scraping)

Run from project root:
    python create_indexes.py

Safe to re-run — MongoDB ignores index creation if index already exists.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient, TEXT, ASCENDING
from config.settings import settings


def create_indexes():
    client = MongoClient(settings.mongodb_uri, tls=True)
    db = client[settings.mongodb_database]
    posts = db["posts"]

    print(f"\n{'─'*55}")
    print(f"  Creating indexes on: {settings.mongodb_database}.posts")
    print(f"{'─'*55}\n")

    existing = {idx["name"] for idx in posts.list_indexes()}
    print(f"  Existing indexes: {existing}\n")

    # ── Index 1: Full-text search on title + content ────────────────────────
    # Allows: db.posts.find({"$text": {"$search": "job rejection"}})
    # Ranks results by relevance score automatically.
    # Without this, search falls back to slow regex (full collection scan).

    text_index_name = "title_text_content_text"

    if text_index_name not in existing:
        print("  Creating text search index (title + content)...")
        posts.create_index(
            [("title", TEXT), ("content", TEXT)],
            name=text_index_name,
            weights={"title": 10, "content": 1},  # title matches weighted 10x
            default_language="english"
        )
        print("  ✅ Text index created — search now ranks by relevance\n")
    else:
        print("  ✅ Text index already exists — skipping\n")

    # ── Index 2: Unique index on post_id ────────────────────────────────────
    # Reddit assigns each post a unique ID (e.g. "abc123").
    # Without this index, running the scraper twice inserts duplicates.
    # With it, duplicate inserts are silently rejected.
    # Also speeds up lookups by post_id.

    unique_index_name = "post_id_unique"

    if unique_index_name not in existing:
        # Check if there are existing duplicates before creating unique index
        pipeline = [
            {"$group": {"_id": "$post_id", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}}
        ]
        duplicates = list(posts.aggregate(pipeline))

        if duplicates:
            print(f"  ⚠️  Found {len(duplicates)} duplicate post_ids in DB.")
            print(f"  Removing duplicates before creating unique index...")

            removed = 0
            for dup in duplicates:
                pid = dup["_id"]
                if not pid:
                    continue
                # Keep the first, delete the rest
                docs = list(posts.find({"post_id": pid}, {"_id": 1}))
                ids_to_delete = [d["_id"] for d in docs[1:]]
                if ids_to_delete:
                    posts.delete_many({"_id": {"$in": ids_to_delete}})
                    removed += len(ids_to_delete)

            print(f"  ✅ Removed {removed} duplicate posts\n")

        print("  Creating unique index on post_id...")
        posts.create_index(
            [("post_id", ASCENDING)],
            name=unique_index_name,
            unique=True,
            sparse=True          # sparse=True ignores documents where post_id is null
        )
        print("  ✅ Unique index created — scraper will skip duplicate posts\n")
    else:
        print("  ✅ Unique index on post_id already exists — skipping\n")

    # ── Index 3: Performance index for pain point queries ───────────────────
    # Speeds up get_pain_points() which always filters by is_pain_point + score

    perf_index_name = "is_pain_point_score"

    if perf_index_name not in existing:
        print("  Creating performance index (is_pain_point + score)...")
        posts.create_index(
            [("is_pain_point", ASCENDING), ("score", -1)],
            name=perf_index_name
        )
        print("  ✅ Performance index created — pain point queries faster\n")
    else:
        print("  ✅ Performance index already exists — skipping\n")

    # ── Summary ─────────────────────────────────────────────────────────────
    all_indexes = [idx["name"] for idx in posts.list_indexes()]
    print(f"{'─'*55}")
    print(f"  All indexes on posts collection:")
    for name in all_indexes:
        print(f"    • {name}")
    print(f"{'─'*55}")
    print(f"\n  ✅ Done. Restart FastAPI to use the new indexes.\n")

    client.close()


if __name__ == "__main__":
    create_indexes()