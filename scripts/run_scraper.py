from scraper.reddit_scraper import RedditScraper, TARGET_SUBREDDITS

if __name__ == "__main__":
    scraper = RedditScraper()
    scraper.scrape_all_subreddits(TARGET_SUBREDDITS)
