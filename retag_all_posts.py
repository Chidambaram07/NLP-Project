"""
retag_all_posts.py
──────────────────
Re-tags ALL posts using the THREE-LAYER HYBRID detection system:
  Layer 1: Keyword exclusions (fast)
  Layer 2: Keyword detection (fast)
  Layer 3: Semantic embedding similarity (catches unseen vocabulary)

The embedding layer uses all-MiniLM-L6-v2 (already installed)
and compares against 30 prototype pain point titles.
This means posts like "My property manager vanished after the inspection"
get correctly classified even though no keyword matches exist.

NOTE: This runs ~50ms per post due to embeddings.
With 6,536 posts: ~5-8 minutes total. Run once after scraping.

Run from project root:
    python retag_all_posts.py

For keyword-only fast mode (no embeddings):
    python retag_all_posts.py --fast
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pymongo import MongoClient, UpdateOne, DESCENDING
from config.settings import settings
from database.operations import _is_pain_point_post, _detect_category, _get_embedding_model

BATCH_SIZE = 200  # smaller batches since embedding adds latency


def retag_all_posts(use_embeddings: bool = True):
    client = MongoClient(settings.mongodb_uri, tls=True)
    db = client[settings.mongodb_database]
    posts_collection = db["posts"]

    total = posts_collection.count_documents({})
    before_pain = posts_collection.count_documents({"is_pain_point": True})

    mode = "hybrid (keyword + embedding)" if use_embeddings else "keyword-only (fast)"

    print(f"\n{'─'*60}")
    print(f"  Re-tagging: {settings.mongodb_database}.posts")
    print(f"  Mode: {mode}")
    print(f"{'─'*60}")
    print(f"  Total posts            : {total:,}")
    print(f"  Pain points (before)   : {before_pain:,}")

    if use_embeddings:
        print(f"\n  Loading embedding model...")
        model, pain_embs, non_pain_embs = _get_embedding_model()
        if model is None:
            print("  ⚠️  Embedding model failed to load. Switching to keyword-only mode.")
            use_embeddings = False
        else:
            print(f"  ✅ Embedding model loaded — will catch unseen vocabulary patterns")
            print(f"  ℹ️  ~50ms per post with embeddings. Estimated time: {total * 50 / 1000 / 60:.0f}-{total * 100 / 1000 / 60:.0f} minutes")
    print(f"{'─'*60}\n")

    cursor = posts_collection.find(
        {},
        {"_id": 1, "title": 1, "content": 1, "selftext": 1, "author": 1, "subreddit": 1}
    )

    batch = []
    processed = 0
    new_pain_count = 0
    keyword_only = 0
    embedding_caught = 0

    for post in cursor:
        # Run keyword layers first
        kw_result = _is_pain_point_post(post, use_embeddings=False)

        if use_embeddings and not kw_result:
            # Keyword said NO — let embedding have a say
            title = (post.get("title") or "").lower().strip()
            content = (post.get("content") or post.get("selftext") or "").lower()

            # Check exclusions first (don't waste embedding on clearly excluded posts)
            excluded = False
            from database.operations import (EXCLUDE_TITLE_PREFIXES,
                                              EXCLUDE_TITLE_KEYWORDS,
                                              EXCLUDE_AUTHORS,
                                              EXCLUDE_SUBREDDITS)
            if post.get("author") in EXCLUDE_AUTHORS: excluded = True
            if post.get("subreddit") in EXCLUDE_SUBREDDITS: excluded = True
            if not excluded:
                for pfx in EXCLUDE_TITLE_PREFIXES:
                    if pfx and title.startswith(pfx): excluded = True; break
            if not excluded:
                for kw in EXCLUDE_TITLE_KEYWORDS:
                    if kw and kw in title: excluded = True; break

            if not excluded:
                final_result = _is_pain_point_post(post, use_embeddings=True)
                if final_result and not kw_result:
                    embedding_caught += 1
            else:
                final_result = False
        else:
            final_result = kw_result
            if kw_result:
                keyword_only += 1

        category = _detect_category(post)

        update_fields = {
            "is_pain_point": final_result,
            "category": category,
            "retagged": True,
            "detection_method": (
                "embedding" if (final_result and not kw_result)
                else "keyword" if final_result
                else "excluded"
            )
        }
        if not final_result:
            update_fields["pain_signal"] = False

        batch.append(UpdateOne({"_id": post["_id"]}, {"$set": update_fields}))
        if final_result:
            new_pain_count += 1
        processed += 1

        if len(batch) >= BATCH_SIZE:
            posts_collection.bulk_write(batch, ordered=False)
            print(
                f"  ✅ {processed:,}/{total:,} | pain: {new_pain_count} "
                f"(+{embedding_caught} from embeddings)...",
                end="\r"
            )
            batch = []

    if batch:
        posts_collection.bulk_write(batch, ordered=False)

    print(f"\n\n{'─'*60}")
    print(f"  Re-tagging complete!")
    print(f"{'─'*60}")
    print(f"  Total processed        : {processed:,}")
    print(f"  Pain points (before)   : {before_pain:,}")
    print(f"  Pain points (after)    : {new_pain_count:,}")
    delta = new_pain_count - before_pain
    print(f"  Change                 : {'+' if delta >= 0 else ''}{delta:,}")
    if use_embeddings:
        print(f"\n  Detection breakdown:")
        print(f"    Keyword detected   : {keyword_only}")
        print(f"    Embedding caught   : {embedding_caught}  ← new posts that keywords missed!")
        print(f"    Total pain points  : {new_pain_count}")

    print(f"\n  Category breakdown (pain points only):")
    for cat in posts_collection.aggregate([
        {"$match": {"is_pain_point": True}},
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": DESCENDING}}
    ]):
        print(f"    {(cat['_id'] or 'General'):<22} {cat['count']:>5}")

    print(f"\n{'─'*60}")
    print("  ✅ Done. Restart FastAPI to reflect changes.\n")


if __name__ == "__main__":
    fast_mode = "--fast" in sys.argv
    retag_all_posts(use_embeddings=not fast_mode)