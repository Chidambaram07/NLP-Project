import time
import praw
import prawcore
from datetime import datetime
from typing import List, Dict

from config.settings import settings
from database.operations import save_posts
from scraper.keywords import TARGET_SUBREDDITS


class RedditScraper:
    """
    Improved Reddit scraper with:
    1. Rate limit backoff — retries on 429 with exponential delay
    2. Multi-filter scraping — hot + top(week) + top(month) per subreddit
       → ~3x more posts from same subreddit without extra API quota
    3. Deduplication is handled by MongoDB unique index on post_id
       (posts already seen are silently skipped by the upsert in save_posts)
    """

    # Time filters to scrape for top posts (in addition to hot)
    TOP_FILTERS = ["week", "month"]

    # Backoff settings for rate limit handling
    MAX_RETRIES = 3
    BACKOFF_BASE = 5    # seconds — doubles on each retry: 5, 10, 20

    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=settings.reddit_client_id,
            client_secret=settings.reddit_client_secret,
            user_agent=settings.reddit_user_agent,
        )
        print("✅ Reddit API initialized")

    def _fetch_with_backoff(self, generator, label: str) -> List[Dict]:
        """
        Fetch posts from a PRAW generator with exponential backoff retry.
        Handles Reddit's 429 rate limit errors gracefully.
        """
        posts = []
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                for submission in generator:
                    posts.append({
                        "post_id":      submission.id,
                        "subreddit":    submission.subreddit.display_name,
                        "title":        submission.title,
                        "content":      submission.selftext,
                        "author":       str(submission.author) if submission.author else "[deleted]",
                        "score":        submission.score,
                        "upvote_ratio": submission.upvote_ratio,
                        "num_comments": submission.num_comments,
                        "created_utc":  datetime.utcfromtimestamp(submission.created_utc),
                        "url":          f"https://www.reddit.com{submission.permalink}",
                        "scraped_at":   datetime.utcnow(),
                        "source":       "reddit_raw"
                    })
                return posts  # success

            except prawcore.exceptions.TooManyRequests:
                wait = self.BACKOFF_BASE * (2 ** (attempt - 1))  # 5, 10, 20 sec
                print(f"  ⚠️  Rate limited on {label}. Waiting {wait}s (attempt {attempt}/{self.MAX_RETRIES})...")
                time.sleep(wait)

            except prawcore.exceptions.ResponseException as e:
                print(f"  ❌ Reddit API error on {label}: {e}")
                break

            except Exception as e:
                print(f"  ❌ Unexpected error on {label}: {e}")
                break

        return posts  # return whatever was collected before error

    def scrape_subreddit(self, subreddit_name: str, limit: int = None) -> List[Dict]:
        """
        Scrape a single subreddit using multiple feed types:
          - hot posts (most current engagement)
          - top posts this week (recent popular content)
          - top posts this month (broader popular content)

        Because save_posts() uses upsert on post_id, duplicate posts
        across feeds are automatically handled — no double counting.
        Total unique posts ≈ 2-3x what hot alone gives.
        """
        if limit is None:
            limit = settings.max_posts_per_subreddit

        all_posts = []
        subreddit = self.reddit.subreddit(subreddit_name)

        try:
            # Feed 1: Hot posts
            print(f"  📥 Scraping r/{subreddit_name} [hot, limit={limit}]...")
            hot_posts = self._fetch_with_backoff(
                subreddit.hot(limit=limit),
                f"r/{subreddit_name}/hot"
            )
            all_posts.extend(hot_posts)

            # Feed 2 & 3: Top posts (week, month)
            for time_filter in self.TOP_FILTERS:
                print(f"  📥 Scraping r/{subreddit_name} [top/{time_filter}, limit={limit}]...")
                top_posts = self._fetch_with_backoff(
                    subreddit.top(time_filter=time_filter, limit=limit),
                    f"r/{subreddit_name}/top/{time_filter}"
                )
                all_posts.extend(top_posts)

            if all_posts:
                # save_posts() uses upsert — duplicates across feeds are merged
                save_posts(all_posts)
                print(f"  ✅ r/{subreddit_name}: {len(all_posts)} posts processed "
                      f"(duplicates upserted, not double-counted)")
            else:
                print(f"  ⚠️  No posts found in r/{subreddit_name}")

        except Exception as e:
            print(f"  ❌ Error scraping r/{subreddit_name}: {e}")

        return all_posts

    def scrape_all_subreddits(self, subreddits: List[str] = None):
        """
        Scrape all target subreddits.
        Prints a summary of new vs updated posts at the end.
        """
        if subreddits is None:
            subreddits = TARGET_SUBREDDITS

        total_processed = 0
        failed = []

        print(f"\n{'='*55}")
        print(f"  Scraping {len(subreddits)} subreddits")
        print(f"  Feeds per subreddit: hot + top/week + top/month")
        print(f"  Rate limit handling: exponential backoff ({self.MAX_RETRIES} retries)")
        print(f"{'='*55}\n")

        for i, subreddit_name in enumerate(subreddits, 1):
            print(f"\n[{i}/{len(subreddits)}] r/{subreddit_name}")
            try:
                posts = self.scrape_subreddit(subreddit_name)
                total_processed += len(posts)
                # Small delay between subreddits to stay within rate limits
                time.sleep(1)
            except Exception as e:
                print(f"  ❌ Failed: {e}")
                failed.append(subreddit_name)

        print(f"\n{'='*55}")
        print(f"  Scraping complete!")
        print(f"  Subreddits scraped : {len(subreddits) - len(failed)}")
        print(f"  Total posts processed: {total_processed}")
        if failed:
            print(f"  Failed subreddits  : {failed}")
        print(f"  Note: MongoDB upsert handles deduplication —")
        print(f"  'processed' count includes updates to existing posts")
        print(f"{'='*55}\n")