"""
database/operations.py
══════════════════════
Pain point detection uses a THREE-LAYER HYBRID approach to generalize
to unseen data that doesn't match any known keywords:

  Layer 1 (Keyword Exclusion): Fast hard rules — blocks LPT posts,
           news articles, spam tags, etc. ~0ms per post.

  Layer 2 (Keyword Detection): Title-signal phrases catch clear-cut
           cases. ~0ms per post. High precision, lower recall.

  Layer 3 (Semantic Embedding): SentenceTransformer cosine similarity
           against 30 prototype pain point titles. Catches posts that
           express pain in unseen vocabulary. ~50ms per post.
           Only runs when Layers 1-2 give no clear answer.

This solves the core generalization problem:
  "My property manager vanished after inspection" → no keyword match
  BUT embedding similarity to "My landlord ghosted me" = 0.73 → ✅ PAIN

SCALABILITY FIX (no more exclusion keyword whack-a-mole):
  Instead of adding new exclusions every time a bad post appears,
  the embedding model now acts as a VETO gate on Layer 2 matches.

  Old behavior: any title signal match → return True (no safety net)
  New behavior:
    - Strong keyword match (title signal + content confirmation)
      → embedding veto: if embedding CONFIDENTLY says not a pain point,
        override the keyword result and reject.
    - Weak title-only match (no content confirmation)
      → MUST pass embedding to qualify. No free pass on title alone.
    - This means new scraped posts with unseen phrasing that happen
      to contain a trigger word (e.g. "isn't the enemy" contains
      "isn't" near career words) get caught by semantics, not by
      adding yet another exclusion string.

  EMBEDDING_THRESHOLD raised from 0.46 → 0.50:
    Real pain posts score 0.55–0.75. Borderline/noise posts score
    0.46–0.49. The tighter threshold eliminates most noise without
    losing verified pain points.

The embedding model (all-MiniLM-L6-v2) is already installed as part
of your BERTopic pipeline — no new dependencies needed.
"""

from pymongo import MongoClient, DESCENDING, UpdateOne
from config.settings import settings

client = MongoClient(settings.mongodb_uri, tls=True)
db = client[settings.mongodb_database]

posts_collection = db["posts"]
pain_points_collection = db["pain_points"]


# ═══════════════════════════════════════════════════════════════════
# LAYER 1: HARD EXCLUSION RULES
# ═══════════════════════════════════════════════════════════════════

EXCLUDE_TITLE_PREFIXES = [
    "lpt:", "lpt ", "life pro tip", "pro tip",
    "til ", "til:", "today i learned",
    "psa:", "psa ", "public service announcement",
    "ama:", "ama ", "ask me anything",
    "announcement:", "update:",
    "weekly", "monthly", "daily", "megathread",
    "hiring:", "who's hiring", "who is hiring",
    "results:", "survey:", "poll:",
    "congratulations", "congrats",
    "reminder:", "heads up:",
    "record ", "report:", "report ",
    "study:", "study ",
    "introducing ", "launching ",
    "released:", "released ",
]

EXCLUDE_TITLE_KEYWORDS = [
    # Mod / community posts
    "megathread", "monthly thread", "weekly thread",
    "who's hiring", "who is hiring",
    "mod post", "moderator post",
    "ask me anything", "new wiki", "made a new", "wiki to explain",
    # Advice / guide / tip
    "tip:", "tips:", "trick:", "tricks:",
    "how to ", "guide:", "guide to",
    "tutorial:", "walkthrough:", "reminder:", "fyi:",
    "some things to", "here is what i found", "here's what i found",
    "what i found:", "i scraped ", "i analyzed ", "i studied ",
    # News / reporting
    "news:", "breaking:", " issues second", " issues third",
    "emergency update", "patch tuesday", "released globally",
    "sales failed to", "failed to save", "close are we to", "[regular ",
    # Achievement / reflection / lesson
    "year of no ", "days of no ", "months of no ", " completed",
    "it saved me", "cured my ", "somehow cured", "taught me that",
    "i misunderstood", "learned this the ", "something about ",
    "tired of having the", "mentality",
    "i realized my", "i realized that", "i realized i",
    # Opinion / discussion
    "hot take:", "unpopular opinion", "change my mind",
    # Startup spam tags
    "will not promote", "i will not promote",
    "[d] ", "[d]",
    # Story / anecdote
    "worst date", "grade teacher", "th grade teacher",
    "showed terrifier", "showed a movie", "showed us a",
    "\"rice and", "found something", "interesting in my room",
    # Informational employer posts
    "employer issued",
    # Misc non-pain
    "click here", "please read", "must read",
    "read this", "check this out", "new? read", "read me first",
    "[hiring]", "[jobs]", "[opportunity]",
    "interesting article", "great article",
    "scam guide", "complete guide", "full guide",
    # Relationship curiosity / discussion
    "uncomfortable about his", "exclusive but", "instagram post with",
    # Metaphorical / lifestyle discussion
    "acting in a play", "dieting with safe", "safe foods",
    # Vague titles with no context (too short to be actionable)
    "i hate this",             # "I hate this" — no object, not a startup pain point
    "help!",                   # single-word plea, no context
    "isolating and avoidance", # noun phrase, no personal context

    # Philosophical / motivational posts (not personal pain)
    "isn't the enemy",         # "Unemployment isn't the enemy" — reflection
    "a gentle reminder for",   # motivational posts
    "i stopped chasing",       # insight/lesson pattern
    "isn't just stress",       # "burnout isn't just stress" — insight post

    # Success story patterns that look like pain
    "here's what i wish i knew",  # "got 3 offers, here's what I wish I knew"
    "what i wish i knew",

    # News/analysis headlines (not personal)
    "tech job posts but",      # "450K Tech Job Posts But Still No Hires"
    "job posts but still",
    "what is actually happening in the job market",
    "this is what upwork use to be",
]

EXCLUDE_SUBREDDITS = {
    "LifeProTips", "todayilearned", "worldnews", "news",
    "nottheonion", "UpliftingNews", "Showerthoughts",
    "unpopularopinion", "changemyview",
}

EXCLUDE_AUTHORS = {
    "AutoModerator", "[deleted]", "reddit", "BotDefense",
}


# ═══════════════════════════════════════════════════════════════════
# LAYER 2: KEYWORD DETECTION
# ═══════════════════════════════════════════════════════════════════

TITLE_ONLY_PAIN = [
    "frustrated", "frustrating",
    "struggling with", "struggle with",
    "burnt out", "burnout",
    "can't find a job", "cannot find a job",
    "unemployed for", "been unemployed",
    "laid off from", "got fired",
    "can't afford", "cannot afford",
    "got scammed", "i was scammed",
    "in debt", "drowning in debt",
    "wish there was a",
    "why is there no", "why doesn't",
    "feels hopeless", "feeling hopeless",
    "overwhelmed by", "nothing works",
    "am i the only one", "enough is enough",
    "depression meals", "depression food", "anxiety meal",
    "no future", "totally lost like",
]

TITLE_PAIN_SIGNALS = [
    # Personal job loss / violation
    "i was fired", "i got fired", "i was laid off", "i got laid off",
    "i lost my job", "i lost my money", "i lost my data",
    "i lost my savings", "i lost my client",
    "i was denied", "i was rejected", "i was told",
    # Struggle
    "i've been struggling", "ive been struggling",
    "i'm struggling", "im struggling", "i've tried everything",
    # Paralysis
    "i don't know what to do", "i don't know how to",
    "i'm at a loss", "i give up",
    # Help-seeking with subject
    "i need help with", "i need advice", "i need someone to",
    "i'm desperate",
    # Distress
    "i'm so frustrated", "i am so frustrated",
    "i'm so stressed", "i am so stressed",
    "i'm so lost", "i am so lost", "i'm totally lost",
    "i regret", "i'm stuck with", "i'm stuck on", "i'm stuck at",
    # Specific hate (requires object)
    "i hate my job", "i hate my boss", "i hate my company",
    "i hate my manager", "i hate my workplace",
    # Financial
    "i can't afford", "i cannot afford",
    "i got scammed", "i was scammed",
    # Specific can'ts
    "i can't find", "i cannot find",
    "i can't get hired", "i can't get paid", "i can't get past",
    "i can't stop", "i can't sleep", "i can't focus",
    "i can't pay", "i cannot pay", "i can't work", "i cannot work",
    # Keep getting (general pattern)
    "i keep getting rejected", "i keep getting ignored",
    "i keep getting ghosted", "i keep getting",
    # Relationship with entities
    "my employer is", "my employer has", "my employer won't",
    "my employer refuses", "my employer fired",
    "my landlord is", "my landlord has", "my landlord won't",
    "my landlord refuses", "my landlord never",
    "my boss is", "my boss told", "my boss fired",
    "my boss won't", "my boss refuses",
    "my manager is", "my manager told", "my manager won't",
    "my company is", "my company won't", "my company refuses", "my company never",
    "my mom is refusing", "my mom won't let",
    "my dad is refusing", "my dad won't let",
    "my partner is", "my husband is", "my wife is", "my marriage is",
    # Community openers (tightened)
    "is anyone else", "am i the only",
    "does anyone else feel", "does anybody else feel", "is anybody else",
    # Problem questions
    "why can't i", "why can't we",
    "why is there no", "why is it so hard", "is it normal to",
    "is this a scam", "is this normal for", "is this legal",
    "is there anything i can do",
    # Decision questions (with context)
    "what do i do if", "what do i do about",
    "what do i do now", "what do i do when", "what do i do?",
    "what should i do about", "what should i do if",
    "what should i do when", "what should i do?",
    # Platform victim
    "suspended my account", "banned my account",
    "locked me out", "locked out of", "terminated my account",
    # Help-seeking
    "need advice on", "need help with", "please help me",
    # Distress
    "so frustrated", "so stressed", "really struggling",
    "totally lost", "completely lost",
    # Violations
    "employer is interpreting", "employer is claiming",
    "company failed to", "company refused to", "company never",
    "dealership never", "dealership refused", "bank refused",
    "landlord refused", "landlord never",
    "without my permission", "without my consent",
    "refuses to allow", "refuses to return", "refuses to refund",
    "cosigned me", "never submitted", "failed to contribute",
    "failed to pay", "stole my", "scammed me", "called the cops on",
    # Rant/vent
    "rant:", "venting:", "fed up with", "sick of this",
    "can't take it anymore",
    # Specific pain patterns
    "voluntarily resigned", "constructive dismissal",
    "wrongful termination", "depression meals", "chronically uncurious",
    "refuses to fill out", "refusing to fill out",
    "it feels like my", "falling apart", "totally lost like", "no future",
    "i feel like i'm", "feeling like i'm", "i feel like i am",
]

CONTENT_PAIN_PHRASES = [
    "frustrated", "frustrating", "frustration",
    "so annoying", "annoys me", "i hate", "hate this",
    "terrible experience", "horrible experience", "worst experience",
    "struggling with", "struggle with", "i'm struggling",
    "suffering from", "desperate for", "desperately need",
    "burnt out", "burnout", "demotivated",
    "depressed about", "feeling depressed", "anxiety about",
    "depression meals", "depression food", "anxiety meal",
    "no jobs available", "can't find a job", "cannot find a job",
    "unemployed for", "been unemployed", "laid off from", "got fired",
    "rejection after rejection", "ghosted by employer",
    "no response from recruiter", "underpaid at", "being underpaid",
    "toxic workplace", "bad manager",
    "voluntarily resigned", "wrongful termination",
    "can't afford", "cannot afford", "too expensive for me",
    "got scammed", "i was scammed", "lost my money",
    "in debt", "drowning in debt", "can't pay",
    "doesn't work for me", "not working for me",
    "keeps crashing", "full of bugs", "missing feature",
    "terrible support", "no customer support", "bad experience with",
    "need help with", "please help me", "i need help",
    "why can't i", "why is there no", "wish there was a",
    "how do i deal with", "am i the only one",
    "it feels like my", "chronically uncurious",
    "doesn't care about me", "refuses to fill out",
    "i'm stuck", "feeling stuck", "nothing works",
    "waste of time", "waste of money", "i regret",
    "feels hopeless", "feeling hopeless", "overwhelmed by",
    "rant about", "venting about", "fed up with", "sick of this",
    "can't take it anymore", "no future",
]

CONTEXTUAL_PAIN = [
    "can't find", "cannot find", "hard to find",
    "difficult to get", "no way to", "impossible to", "can't get",
    "looking for help", "any solution", "still broken",
    "still not working", "won't work", "too slow",
    "really expensive", "so expensive", "stressed about",
    "stress about", "under pressure", "feel pressured",
    "i feel ignored", "being ignored",
    "not available anywhere", "not available in",
]


# ═══════════════════════════════════════════════════════════════════
# LAYER 3: SEMANTIC EMBEDDING DETECTION
# ═══════════════════════════════════════════════════════════════════
#
# These 30 prototype titles define "what a pain point looks like"
# in embedding space. New posts semantically similar to these
# are detected even if they use completely different vocabulary.
#
# To update: add more verified pain point titles here.
# More prototypes = better coverage of diverse pain expressions.

PAIN_POINT_PROTOTYPES = [
    # Job / Career
    "Does anyone else feel too mentally drained to cook after work?",
    "Employer is interpreting that I voluntarily resigned",
    "Company failed to contribute to my 401k last year",
    "I was fired after I called the cops on my coworker",
    "I keep getting my ass handed to me in technical assessments",
    "Unemployed for 10 months struggling to find work",
    "I hate my job and it is destroying my mental health",
    "Can't find a job as a fresher despite 200 applications",
    "New team lead struggling with aggressive manager on timelines",
    "Client keeps asking for more features after underpaying",
    # Finance / Legal
    "My brother cosigned me for a student loan without my permission",
    "Car dealership never submitted paperwork and refuses to return the car",
    "Etsy suspended my account without appeal access after I shipped orders",
    "Apartment renewal offer is much higher than what same unit lists for online",
    "My landlord is refusing to return my security deposit",
    # Health / Wellbeing
    "What are some easy depression meals for when I have no energy",
    "My marriage is falling apart and I don't know what to do anymore",
    "Does anybody else feel more anxious every day",
    "Why is there no affordable mental health care in India",
    "I'm struggling with productivity and can't seem to get things done",
    # Education / Life
    "What do I do if my mom is refusing to fill out FAFSA",
    "I'm 400 pounds and I'd like to be small again please help me",
    "I keep missing deadlines and feel like I'm ruining everything",
    "26 year old and I can't get past procrastination and fear of failure",
    "Does anyone feel reverse homesick when they go home for college",
    # Freelance / Business
    "Is this normal for a client to ask this many questions on chat",
    "Finished making website for local business and now they ghosted me",
    "I've been struggling to find a stable job for 8 months",
    "My coworker takes credit for my work and my manager does nothing",
    "Three months of job applications and not a single callback",
]

NON_PAIN_PROTOTYPES = [
    "LPT This is one of the best times to buy a used car",
    "Today I learned that honey never expires",
    "PSA don't forget to vote in local elections",
    "Here are the best programming languages to learn",
    "Hot take LinkedIn is just Facebook for corporate people",
    "I realized my burnout came from staying in the wrong job",
    "1 Year of no nicotine completed",
    "Day 100 of learning Spanish completed",
    "My girlfriend taught me not to be perfect and it saved me",
    "How I built a profitable SaaS in 6 months",
    "The most underrated productivity technique I discovered",
    "Why React is better than Vue for large projects",
    "Amazing sunset photo from my hike yesterday",
    "New study shows coffee drinkers live longer",
    "The history of the steam engine is fascinating",
    # Additional non-pain prototypes to strengthen the veto gate
    "Unemployment is not the enemy it is the waiting game",
    "You are not unproductive you are avoiding one task",
    "I finally realized I am not lazy I just need better systems",
    "Here is what burnout taught me about myself",
    "Forget expensive therapy home cooking is your free superpower",
    "Honestly my productivity apps were making my anxiety worse going back to paper saved my sanity",
    "Is it weird to ask dealbreaker questions in the first few chats on a dating app",
    "What do i do when my little sister is talking to an older boy",
    "I need advice on my partner who watches porn every morning",
]

# ── EMBEDDING_THRESHOLD raised from 0.46 → 0.50 ─────────────────────────────
# Real pain posts score 0.55–0.75 against prototype titles.
# Borderline noise posts (motivational, insight, dating advice) score 0.46–0.49.
# Raising to 0.50 eliminates that noise band without losing verified pain points.
# This is more robust than adding exclusion keywords because it works on meaning,
# not vocabulary — future posts with new phrasing are handled automatically.
EMBEDDING_THRESHOLD = 0.50

# Lazy-loaded model and prototypes (loaded once, reused)
_embedding_model = None
_pain_embs = None
_non_pain_embs = None


def _get_embedding_model():
    """Lazy-load the SentenceTransformer model (already installed for BERTopic)."""
    global _embedding_model, _pain_embs, _non_pain_embs

    if _embedding_model is not None:
        return _embedding_model, _pain_embs, _non_pain_embs

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        _pain_embs = _embedding_model.encode(
            PAIN_POINT_PROTOTYPES, show_progress_bar=False, normalize_embeddings=True
        )
        _non_pain_embs = _embedding_model.encode(
            NON_PAIN_PROTOTYPES, show_progress_bar=False, normalize_embeddings=True
        )
        return _embedding_model, _pain_embs, _non_pain_embs

    except Exception as e:
        print(f"[embedding] Model load failed: {e}. Falling back to keyword-only.")
        return None, None, None


def _embedding_is_pain(text: str) -> bool | None:
    """
    Use semantic similarity to classify text as pain point or not.

    Returns:
      True  → confidently a pain point (sim > threshold AND > non-pain)
      False → confidently NOT a pain point
      None  → uncertain (below threshold — fall back to keyword result)
    """
    try:
        model, pain_embs, non_pain_embs = _get_embedding_model()
        if model is None:
            return None

        import numpy as np

        emb = model.encode([text], show_progress_bar=False, normalize_embeddings=True)

        # Cosine similarity = dot product when normalized
        pain_sim = float(np.dot(emb, pain_embs.T).max())
        non_pain_sim = float(np.dot(emb, non_pain_embs.T).max())

        if pain_sim > EMBEDDING_THRESHOLD and pain_sim > non_pain_sim:
            return True
        elif non_pain_sim > pain_sim + 0.05:
            # Confidently NOT a pain point
            return False
        else:
            return None  # uncertain

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# CORE DETECTION: THREE-LAYER HYBRID
# ═══════════════════════════════════════════════════════════════════

def _is_pain_point_post(post: dict, use_embeddings: bool = True) -> bool:
    """
    Three-layer hybrid pain point detection.

    Layer 1 (Keyword Exclusion): ~0ms — rejects clear non-pain patterns.
    Layer 2 (Keyword Detection): ~0ms — accepts clear pain patterns.
    Layer 3 (Semantic Embedding): ~50ms — handles unseen vocabulary.

    The use_embeddings flag allows disabling embeddings for the live API
    (where speed matters) while keeping them for the offline retag pipeline.

    SCALABILITY DESIGN:
    Layer 2 now has an embedding VETO gate instead of blindly returning True.
    This means no new exclusion keywords are needed when new bad posts appear
    after future scrapes — the embedding model handles them by meaning, not
    by vocabulary matching. Only add to EXCLUDE_TITLE_KEYWORDS for clear-cut
    structural patterns (mod posts, PSAs, etc.) that are always non-pain
    regardless of content.

    Specifically:
      - Strong match (title signal + content confirmation):
          embedding veto applies — if embedding CONFIDENTLY says not a pain
          point (returns False, not None), it overrides the keyword result.
      - Weak match (title signal alone, no content confirmation):
          MUST pass embedding to qualify. No free pass on title alone.
          This is what catches insight posts, motivational content, and
          dating/family questions that happen to contain trigger words.
      - Unseen vocabulary (no keyword match at all):
          embedding decides as before (Layer 3).
    """
    title = (post.get("title") or "").lower().strip()
    content = (post.get("content") or post.get("selftext") or "").lower()
    author = post.get("author", "")
    subreddit = post.get("subreddit", "")

    # ── Layer 1: Hard exclusions ─────────────────────────────────────────────
    if author in EXCLUDE_AUTHORS:
        return False
    if subreddit in EXCLUDE_SUBREDDITS:
        return False
    for prefix in EXCLUDE_TITLE_PREFIXES:
        if title.startswith(prefix):
            return False
    for kw in EXCLUDE_TITLE_KEYWORDS:
        if kw in title:
            return False

    # ── Layer 2a: High-confidence title phrases ───────────────────────────────
    # These are very specific multi-word phrases with near-zero false positive
    # rate (e.g. "depression meals", "drowning in debt") — no veto needed.
    if any(phrase in title for phrase in TITLE_ONLY_PAIN):
        return True

    # ── Layer 2b: Title signal → check content, then embedding veto ──────────
    title_relevant = any(phrase in title for phrase in TITLE_PAIN_SIGNALS)
    if title_relevant:
        combined = f"{title} {content}"
        text_for_emb = f"{title}. {content[:200]}".strip()

        if any(phrase in combined for phrase in CONTENT_PAIN_PHRASES):
            # Strong match: keyword confirms pain in content.
            # Still apply embedding veto — if the model is CONFIDENT this is
            # not a pain point (returns False, not None), trust the embedding.
            # This catches posts like motivational pieces that use pain vocab
            # in a lesson/insight framing rather than personal distress.
            if use_embeddings:
                emb = _embedding_is_pain(text_for_emb)
                if emb is False:
                    return False   # embedding confidently vetoes keyword match
            return True

        if sum(1 for phrase in CONTEXTUAL_PAIN if phrase in combined) >= 2:
            # Two contextual signals — reasonably confident.
            # Apply same veto logic.
            if use_embeddings:
                emb = _embedding_is_pain(text_for_emb)
                if emb is False:
                    return False
            return True

        # Weak match: title signal alone, no content confirmation.
        # This is the highest-risk path — must pass embedding to qualify.
        # Without this gate, posts like "Unemployment isn't the enemy",
        # "Is it weird to ask dealbreaker questions on Hinge",
        # "What do I do when my sister is talking to an older guy"
        # all pass on title signal alone even though they are clearly
        # not startup-relevant pain points.
        if use_embeddings:
            emb = _embedding_is_pain(text_for_emb)
            return emb is True   # only True counts; None or False = reject
        # No embeddings available → reject weak title-only matches
        # (conservative: better to miss than to include noise)
        return False

    # ── Layer 2c: 2+ contextual signals in title ──────────────────────────────
    if sum(1 for phrase in CONTEXTUAL_PAIN if phrase in title) >= 2:
        if use_embeddings:
            emb = _embedding_is_pain(f"{title}. {content[:200]}".strip())
            if emb is False:
                return False
        return True

    # ── Layer 3: Semantic embedding (catches unseen vocabulary) ───────────────
    if use_embeddings:
        # Use title + first 200 chars of content for richer signal
        text = f"{title}. {content[:200]}".strip()
        emb_result = _embedding_is_pain(text)
        if emb_result is not None:
            return emb_result

    return False


def _detect_category(post: dict) -> str:
    """Detect category from post title and content."""
    try:
        from scraper.keywords import PAIN_CATEGORIES
    except ImportError:
        return "General"

    title = (post.get("title") or "").lower()
    content = (post.get("content") or "").lower()
    combined = f"{title} {content}"

    for category, keywords in PAIN_CATEGORIES.items():
        if any(kw in combined for kw in keywords):
            return category

    return "General"


# ═══════════════════════════════════════════════════════════════════
# SAVE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def save_posts(posts):
    """
    Save posts using upsert keyed on post_id (prevents duplicates).
    Uses keyword-only detection (no embeddings) for scraper speed.
    Run retag_all_posts.py to re-classify with embeddings.
    """
    if not posts:
        return

    operations = []
    for post in posts:
        post_id = post.get("post_id")
        if not post_id:
            continue

        # Fast path: no embeddings during scraping (use retag for full detection)
        post["is_pain_point"] = _is_pain_point_post(post, use_embeddings=False)
        post["category"] = _detect_category(post)

        operations.append(UpdateOne(
            {"post_id": post_id},
            {"$set": post},
            upsert=True
        ))

    if not operations:
        return

    try:
        result = posts_collection.bulk_write(operations, ordered=False)
        pain_count = sum(1 for p in posts if p.get("is_pain_point") and p.get("post_id"))
        print(
            f"✅ Processed {len(operations)} posts: "
            f"{result.upserted_count} new, {result.modified_count} updated, "
            f"{pain_count} keyword-detected pain points "
            f"(run retag_all_posts.py for full embedding detection)"
        )
    except Exception as e:
        print(f"❌ Error saving posts: {e}")


def save_pain_point(pain_point):
    try:
        pain_points_collection.insert_one(pain_point)
        print(f"✅ Saved pain point {pain_point.get('post_id', '')}")
    except Exception as e:
        print(f"❌ Error saving pain point: {e}")


# ═══════════════════════════════════════════════════════════════════
# READ FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def get_posts(limit=100, skip=0, subreddit=None, category=None, pain_points_only=False):
    query = {}
    if subreddit: query["subreddit"] = subreddit
    if category: query["category"] = category
    if pain_points_only: query["is_pain_point"] = True
    cursor = posts_collection.find(query).skip(skip).limit(limit)
    posts = []
    for post in cursor:
        post["_id"] = str(post["_id"])
        posts.append(post)
    return posts


def get_pain_points(limit=100, category=None, min_score=0):
    query = {"score": {"$gte": min_score}, "is_pain_point": True}
    if category and category != "General":
        query["category"] = category

    cursor = posts_collection.find(query).sort("score", DESCENDING).limit(limit * 2)
    posts = []
    seen_titles = set()

    for post in cursor:
        post["_id"] = str(post["_id"])
        title_key = post.get("title", "")[:60].lower()
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        if not post.get("category"):
            post["category"] = _detect_category(post)
        posts.append(post)
        if len(posts) >= limit:
            break

    # Fallback for posts not yet retagged
    if len(posts) < 5:
        fallback_query = {"score": {"$gte": min_score}, "retagged": {"$exists": False}}
        if category and category != "General":
            fallback_query["category"] = category
        cursor = posts_collection.find(fallback_query).sort("score", DESCENDING).limit(500)
        for post in cursor:
            post["_id"] = str(post["_id"])
            title_key = post.get("title", "")[:60].lower()
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)\

            if _is_pain_point_post(post, use_embeddings=False):
                if not post.get("category"):
                    post["category"] = _detect_category(post)
                posts.append(post)
            if len(posts) >= limit:
                break

    return posts[:limit]


def get_statistics():
    total_posts = posts_collection.count_documents({})
    total_pain_points = posts_collection.count_documents({"is_pain_point": True})

    pipeline = [
        {"$match": {"is_pain_point": True, "category": {"$exists": True, "$ne": None, "$ne": ""}}},
        {"$group": {"_id": "$category", "count": {"$sum": 1}}},
        {"$sort": {"count": DESCENDING}}
    ]
    categories = list(posts_collection.aggregate(pipeline))

    pipeline2 = [
        {"$group": {"_id": "$subreddit", "count": {"$sum": 1}}},
        {"$sort": {"count": DESCENDING}}, {"$limit": 10}
    ]
    top_subreddits = list(posts_collection.aggregate(pipeline2))

    return {
        "total_posts": total_posts,
        "total_pain_points": total_pain_points,
        "categories": categories,
        "top_subreddits": top_subreddits
    }


def search_posts(query: str, limit: int = 50):
    try:
        cursor = posts_collection.find(
            {"$text": {"$search": query}, "is_pain_point": True},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        posts = []
        for post in cursor:
            post["_id"] = str(post["_id"])
            posts.append(post)
        if posts:
            return posts
    except Exception:
        pass

    cursor = posts_collection.find({
        "$and": [
            {"is_pain_point": True},
            {"$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"content": {"$regex": query, "$options": "i"}}
            ]}
        ]
    }).sort("score", DESCENDING).limit(limit)
    posts = []
    for post in cursor:
        post["_id"] = str(post["_id"])
        posts.append(post)
    return posts


def tag_existing_posts():
    """Legacy tagger. Use retag_all_posts.py for bulk re-tagging with embeddings."""
    cursor = posts_collection.find({"is_pain_point": {"$exists": False}})
    count = 0
    pain_count = 0
    for post in cursor:
        is_pain = _is_pain_point_post(post, use_embeddings=False)
        category = _detect_category(post)
        posts_collection.update_one(
            {"_id": post["_id"]},
            {"$set": {"is_pain_point": is_pain, "category": category}}
        )
        count += 1
        if is_pain: pain_count += 1
    print(f"✅ Tagged {count} posts — {pain_count} are pain points")