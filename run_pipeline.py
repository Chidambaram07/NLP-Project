"""
run_pipeline.py — NLP Opportunity Analysis Pipeline
====================================================
Runs offline NLP analysis on verified pain point posts:
  1. Sentiment analysis (DistilBERT)
  2. Topic modeling (BERTopic + KMeans)
  3. Trend analysis (log-linear regression)
  4. Opportunity scoring (weighted composite)
  5. Writes results back to MongoDB

FIX vs original: Loads posts by is_pain_point=True (the new
verified field) instead of the old preprocessed/is_candidate
fields which were set by the previous weak detection logic.

FIX competition score: Was hardcoded at 0.5 for every topic.
Now computed as inverse normalized volume — topics with more posts
signal a more crowded space (higher competition = lower whitespace).

FIX topic_keywords write-back: Now stored on each post document
so /solutions/generate can include semantic topic context in the
Groq prompt via get_nlp_context() in routes.py.

Run from project root:
    python run_pipeline.py
"""

from datetime import datetime
from collections import defaultdict, Counter
from typing import List

from config.database import db
from nlp_engine.sentiment import analyze_sentiment
from nlp_engine.topic_model import run_topic_modeling
from nlp_engine.trend_analysis import analyze_trends
from nlp_engine.scoring import compute_opportunity_scores


POSTS_COLLECTION = db["posts"]

# Minimum posts required for topic modeling to produce meaningful results
MIN_POSTS_FOR_TOPICS = 30


def load_pain_point_posts(limit: int = 500) -> List[dict]:
    """
    Load verified pain point posts for NLP analysis.

    Uses is_pain_point=True (set by retag_all_posts.py) rather than the
    old preprocessed/is_candidate fields from preprocess_reddit.py, which
    used weak keyword logic and is no longer the source of truth.

    Falls back to processed_text if available; otherwise uses title+content.
    """
    cursor = POSTS_COLLECTION.find(
        {"is_pain_point": True},
        {
            "_id": 1,
            "title": 1,
            "content": 1,
            "selftext": 1,
            "processed_text": 1,
            "clean_text": 1,
            "created_utc": 1,
            "category": 1,
            "subreddit": 1,
            "score": 1,
        }
    ).sort("score", -1).limit(limit)

    posts = list(cursor)
    print(f"✅ Loaded {len(posts)} verified pain point posts")
    return posts


def get_text_for_analysis(post: dict) -> str:
    """
    Get the best available text for NLP analysis.
    Preference order: processed_text > clean_text > title + content
    """
    if post.get("processed_text") and len(post["processed_text"].strip()) > 10:
        return post["processed_text"]
    if post.get("clean_text") and len(post["clean_text"].strip()) > 10:
        return post["clean_text"]
    # Fallback: combine title and content, basic clean
    import re
    title = post.get("title", "")
    content = post.get("content") or post.get("selftext") or ""
    combined = f"{title}. {content}"
    combined = re.sub(r'https?://\S+', '', combined)
    combined = re.sub(r'\*{1,3}|\[.*?\]\(.*?\)', '', combined)
    combined = re.sub(r'\s+', ' ', combined).strip().lower()
    return combined


def main():
    print("\n" + "="*60)
    print("  NLP Opportunity Analysis Pipeline")
    print("="*60 + "\n")

    # ── Load data ──────────────────────────────────────────────
    posts = load_pain_point_posts(limit=500)

    if len(posts) < MIN_POSTS_FOR_TOPICS:
        print(f"⚠️  Only {len(posts)} posts loaded — topic modeling works best")
        print(f"   with {MIN_POSTS_FOR_TOPICS}+ posts. Results may be sparse.")
        print(f"   Tip: Scrape more data or lower min_score in get_pain_points()\n")

    if not posts:
        print("❌ No pain point posts found. Run the scraper first.")
        return

    # ── Sample check ───────────────────────────────────────────
    print("\n── Sample posts ──")
    for i, p in enumerate(posts[:3]):
        text = get_text_for_analysis(p)
        print(f"  [{i+1}] {p.get('title', '')[:70]}")
        print(f"       text: {text[:80]}...")
        print()

    # ── Prepare texts and timestamps ───────────────────────────
    texts = [get_text_for_analysis(p) for p in posts]
    timestamps = [p.get("created_utc", datetime.utcnow()) for p in posts]

    # Filter out empty texts
    valid = [(t, ts, p) for t, ts, p in zip(texts, timestamps, posts) if t.strip()]
    if len(valid) < len(posts):
        print(f"ℹ️  Skipped {len(posts) - len(valid)} posts with empty text")
    texts, timestamps, posts = zip(*valid) if valid else ([], [], [])
    texts, timestamps, posts = list(texts), list(timestamps), list(posts)

    if not texts:
        print("❌ No valid text found for analysis.")
        return

    # ── Sentiment Analysis ─────────────────────────────────────
    print(f"── Running sentiment analysis on {len(texts)} posts...")
    print("   (DistilBERT loads in ~20 seconds on first run — please wait)\n")
    sentiments = []
    for i, t in enumerate(texts):
        sent = analyze_sentiment(t)
        sentiments.append(sent)
        if (i + 1) % 50 == 0:
            print(f"   Sentiment: {i+1}/{len(texts)} done...")
    print(f"   ✅ Sentiment analysis complete\n")

    # ── Topic Modeling ─────────────────────────────────────────
    n_topics = min(12, max(3, len(texts) // 10))   # scale topics to data size
    print(f"── Running topic modeling (BERTopic, {n_topics} topics)...")
    topics, topic_keywords = run_topic_modeling(texts, n_topics=n_topics)
    print(f"   ✅ Topic modeling complete\n")

    # ── Trend Analysis ─────────────────────────────────────────
    print("── Analyzing topic trends...")
    trend_scores = analyze_trends(topics, timestamps)
    print(f"   ✅ Trend analysis complete\n")

    # ── Aggregate per-topic stats ──────────────────────────────
    topic_agg = defaultdict(lambda: {
        "count": 0,
        "sentiment_sum": 0.0,
        "sentiment_labels": [],
        "categories": []
    })

    for post, topic, sent in zip(posts, topics, sentiments):
        if topic == -1:
            continue
        topic_agg[topic]["count"] += 1
        topic_agg[topic]["sentiment_sum"] += sent["compound"]
        topic_agg[topic]["sentiment_labels"].append(sent["label"])
        topic_agg[topic]["categories"].append(post.get("category", "General"))

    # ── FIX: Compute competition as inverse normalized volume ──
    # More posts in a topic = more people talking about it = more
    # existing solutions = higher competition = lower whitespace score.
    # This replaces the hardcoded 0.5 that was used for every topic.
    topic_counts = {t: s["count"] for t, s in topic_agg.items()}
    max_count = max(topic_counts.values(), default=1)
    min_count = min(topic_counts.values(), default=0)
    count_range = max(max_count - min_count, 1)

    competition_scores = {
        tid: round((cnt - min_count) / count_range, 3)
        for tid, cnt in topic_counts.items()
    }

    print(f"── Competition scores (inverse volume):")
    for tid, comp in sorted(competition_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   Topic {tid}: {comp:.3f} ({topic_counts[tid]} posts)")
    print()

    # ── Build topic_stats ──────────────────────────────────────
    topic_stats = {}
    for topic, stats in topic_agg.items():
        avg_sentiment = stats["sentiment_sum"] / stats["count"]
        dominant_sentiment = Counter(stats["sentiment_labels"]).most_common(1)[0][0]
        dominant_category = Counter(stats["categories"]).most_common(1)[0][0]

        topic_stats[topic] = {
            "demand": stats["count"],
            "sentiment": avg_sentiment,
            "sentiment_label": dominant_sentiment,
            "dominant_category": dominant_category,
            "trend": trend_scores.get(topic, 0.0),
            "competition": competition_scores.get(topic, 0.5)  # FIX: no longer hardcoded
        }

    # ── Compute Opportunity Scores ─────────────────────────────
    print("── Computing opportunity scores...")
    scores = compute_opportunity_scores(topic_stats)
    print(f"   ✅ Scoring complete\n")

    # ── Print results ──────────────────────────────────────────
    opportunities = []
    for topic, score in scores.items():
        opportunities.append({
            "topic": topic,
            "score": score,
            "volume": topic_stats[topic]["demand"],
            "trend": topic_stats[topic]["trend"],
            "sentiment": topic_stats[topic]["sentiment_label"],
            "category": topic_stats[topic]["dominant_category"],
            "competition": topic_stats[topic]["competition"],
            "keywords": topic_keywords.get(topic, [])[:5]   # top 5 keywords
        })

    opportunities.sort(key=lambda x: x["score"], reverse=True)

    print("="*60)
    print("  TOP OPPORTUNITIES")
    print("="*60)
    for i, opp in enumerate(opportunities[:10], 1):
        kw = [k[0] if isinstance(k, (list, tuple)) else k for k in opp["keywords"]]
        print(f"\n  #{i} Topic {opp['topic']} — Score: {opp['score']:.4f}")
        print(f"     Category   : {opp['category']}")
        print(f"     Volume     : {opp['volume']} posts")
        print(f"     Trend      : {opp['trend']:+.3f}")
        print(f"     Sentiment  : {opp['sentiment']}")
        print(f"     Competition: {opp['competition']:.3f}")
        print(f"     Keywords   : {', '.join(str(k) for k in kw)}")

    # ── Write results back to MongoDB ──────────────────────────
    # FIX: now also writes topic_keywords to each post so that
    # get_nlp_context() in routes.py can include them in Groq prompts.
    print(f"\n── Writing NLP results to MongoDB...")
    updated = 0
    for post, sent, topic in zip(posts, sentiments, topics):
        if topic == -1:
            continue

        # Get keywords for this post's topic (store top 8)
        post_topic_kws = topic_keywords.get(topic, [])[:8]

        POSTS_COLLECTION.update_one(
            {"_id": post["_id"]},
            {
                "$set": {
                    "sentiment": {
                        "label": (
                            "positive" if sent["compound"] > 0.05
                            else "negative" if sent["compound"] < -0.05
                            else "neutral"
                        ),
                        "compound": round(sent["compound"], 3),
                        "complaint_intensity": round(sent.get("complaint_intensity", 0), 3),
                    },
                    "topic_id": int(topic),
                    "topic_keywords": post_topic_kws,       # FIX: was missing before
                    "trend": trend_scores.get(topic, 0.0),
                    "opportunity_score": scores.get(topic, 0.0),
                    "competition_score": competition_scores.get(topic, 0.5),
                    "pipeline_version": "v2",
                    "pipeline_ran_at": datetime.utcnow()
                }
            }
        )
        updated += 1

    print(f"   ✅ Updated {updated} posts in MongoDB")
    print(f"\n{'='*60}")
    print(f"  Pipeline complete! {len(opportunities)} opportunity topics found.")
    print(f"  Each post now has: sentiment, topic_id, topic_keywords,")
    print(f"  trend, opportunity_score, competition_score, pipeline_version=v2")
    print(f"  → /solutions/generate will now include NLP signals in every prompt.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()