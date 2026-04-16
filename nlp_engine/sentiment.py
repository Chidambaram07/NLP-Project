"""
nlp_engine/sentiment.py
════════════════════════
Pain-Aware Sentiment Scoring (PASS)
─────────────────────────────────────────────────────────────
Novel Contribution:
  Existing sentiment models (VADER, SST-2, Twitter-RoBERTa) classify
  text as positive/negative/neutral. For entrepreneurial opportunity
  detection, what matters is not sentiment polarity alone but
  COMPLAINT ACTIONABILITY — whether the negative sentiment represents
  an unmet need that a product can address.

  PASS is a weighted ensemble that combines four domain-specific
  components into a single actionability score:

  Component 1 — Twitter-RoBERTa base score (α = 0.40)
    cardiffnlp/twitter-roberta-base-sentiment-latest
    Trained on 200M tweets — social media language, sarcasm, slang,
    Indian English. Far more appropriate than SST-2 (movie reviews)
    for Reddit pain point analysis.
    Citation: Barbieri et al., 2020 (Cardiff NLP)

  Component 2 — Pain lexicon density (β = 0.30)
    Normalized count of domain-specific pain keywords per 100 words.
    A post with 8 pain keywords per 100 words is more actionable
    than one with 2, even if transformer score is identical.
    formula: pain_density = (keyword_hits / word_count) * 100

  Component 3 — Engagement amplification (γ = 0.20)
    Reddit upvotes are crowd-sourced signal of resonance. A post with
    2,258 upvotes about an unmet need represents more market demand
    than a 5-upvote post with identical text. Log-normalised to [0,1].
    formula: engagement_weight = log(score+1) / log(max_score+1)
    Default max_score = 10,000 (typical Reddit pain post ceiling)

  Component 4 — Negation/intensifier correction (δ = 0.10)
    Rule-based correction: intensifiers ("so frustrated", "absolutely
    terrible") amplify the score. Negations ("isn't that bad",
    "actually fine") reduce it. Catches cases where transformer
    confidence is high but framing is non-complaint.

  Final PASS score:
    PASS = α·roberta_neg + β·pain_density_norm + γ·engagement + δ·correction
    Range: [0, 1]. Higher = more actionable pain signal.

  Backward compatible: analyze_sentiment() returns all original fields
  plus new pass_score and roberta_* fields. run_pipeline.py requires
  no changes — it calls analyze_sentiment() and reads compound/label.
"""

import re
import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─────────────────────────────────────────────────────────────
# Component 1: Twitter-RoBERTa (domain-appropriate base model)
# Replaces distilbert-base-uncased-finetuned-sst-2-english
# which was trained on movie reviews — wrong domain for Reddit.
# ─────────────────────────────────────────────────────────────

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
THRESHOLD = 0.6   # τ for opportunity detection (Equation 2)

# Label mapping for twitter-roberta-base-sentiment-latest:
# index 0 = negative, index 1 = neutral, index 2 = positive
# (different from SST-2 which only has 0=neg, 1=pos)
ROBERTA_LABELS = {0: "negative", 1: "neutral", 2: "positive"}

_roberta_tokenizer = None
_roberta_model = None
_device = None


def _load_roberta():
    """Lazy-load Twitter-RoBERTa (only on first call)."""
    global _roberta_tokenizer, _roberta_model, _device

    if _roberta_model is not None:
        return _roberta_tokenizer, _roberta_model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _roberta_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_MODEL)
    _roberta_model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL)
    _roberta_model.to(_device)
    _roberta_model.eval()

    return _roberta_tokenizer, _roberta_model, _device


# ─────────────────────────────────────────────────────────────
# Component 2: Pain Lexicon (domain-specific Reddit pain vocab)
# ─────────────────────────────────────────────────────────────

PAIN_KEYWORDS = {
    # Performance / technical
    "slow", "lag", "crash", "freeze", "glitch", "timeout", "delay",
    "bug", "issue", "problem", "error", "broken", "not working",
    "failure", "fault", "defect",

    # Frustration / emotional distress
    "hate", "annoying", "frustrating", "useless", "terrible",
    "worst", "awful", "disappointed", "hopeless", "desperate",
    "stuck", "lost", "overwhelmed", "exhausted", "drained",
    "burnout", "burnt out", "anxious", "anxiety", "depressed",
    "struggling", "suffering",

    # Unmet need signals
    "need", "wish", "missing", "lack", "improve", "no solution",
    "no alternative", "wish there was", "why is there no",
    "can't find", "cannot find",

    # Financial pain
    "expensive", "overpriced", "costly", "debt", "afford",
    "scammed", "ghosted", "no response", "no support",

    # Job / career pain
    "unemployed", "fired", "laid off", "rejected", "ghosted",
    "underpaid", "toxic", "burnout", "no callback",
}

# ─────────────────────────────────────────────────────────────
# Component 4: Intensifier / Negation patterns
# ─────────────────────────────────────────────────────────────

INTENSIFIERS = [
    "so frustrated", "so stressed", "absolutely terrible",
    "can't take it anymore", "completely lost", "totally lost",
    "really struggling", "desperately need", "nothing works",
    "fed up", "sick of this", "enough is enough", "no future",
    "falling apart", "ruining everything", "drowning in",
]

NEGATIONS = [
    "isn't that bad", "not that bad", "actually fine",
    "isn't the enemy", "not the problem", "solved it",
    "figured it out", "it worked out", "glad i did",
    "best decision", "no longer", "fixed now",
    "it saved me", "taught me", "i realized",
]

# Max reference score for engagement normalisation
# (log scale — 10,000 upvotes normalises to 1.0)
ENGAGEMENT_MAX_SCORE = 10_000

# PASS component weights — must sum to 1.0
PASS_WEIGHTS = {
    "roberta":    0.40,   # α — transformer base score
    "lexicon":    0.30,   # β — pain keyword density
    "engagement": 0.20,   # γ — upvote amplification
    "correction": 0.10,   # δ — intensifier/negation correction
}


# ─────────────────────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────────
# Component implementations
# ─────────────────────────────────────────────────────────────

def _roberta_scores(text: str) -> dict:
    """
    Component 1: Run Twitter-RoBERTa and return per-class probabilities.
    Returns dict with keys: negative, neutral, positive (all float [0,1])
    """
    tokenizer, model, device = _load_roberta()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)[0]

    return {
        "negative": probs[0].item(),
        "neutral":  probs[1].item(),
        "positive": probs[2].item(),
    }


def _pain_lexicon_density(text: str) -> float:
    """
    Component 2: Normalized pain keyword density.
    Returns float [0, 1] — higher means denser pain signal.

    Formula: pain_density = min(1.0, (keyword_hits / word_count) * 10)
    Multiplier of 10 maps typical pain post density (~0.08) to ~0.8.
    """
    words = text.split()
    word_count = max(len(words), 1)
    keyword_hits = sum(1 for kw in PAIN_KEYWORDS if kw in text)
    raw_density = (keyword_hits / word_count) * 10
    return round(min(1.0, raw_density), 4)


def _engagement_weight(upvote_score: int) -> float:
    """
    Component 3: Log-normalised Reddit upvote amplification.
    Returns float [0, 1].

    Formula: log(score+1) / log(MAX_SCORE+1)
    A post with 0 upvotes gets 0.0. A post with ENGAGEMENT_MAX_SCORE
    upvotes gets 1.0. Log scale prevents viral posts from dominating.

    Default call (no upvote data) uses score=0 → returns 0.0,
    meaning engagement does not influence the score. This is correct
    behaviour when called from run_pipeline.py without upvote context.
    """
    if upvote_score <= 0:
        return 0.0
    weight = math.log(upvote_score + 1) / math.log(ENGAGEMENT_MAX_SCORE + 1)
    return round(min(1.0, weight), 4)


def _intensifier_correction(text: str) -> float:
    """
    Component 4: Rule-based intensifier/negation correction.
    Returns float [-0.3, +0.3].

    Intensifiers push the score up (more actionable pain).
    Negations pull the score down (post is framed as resolution/insight,
    not active complaint — even if it contains pain keywords).
    """
    correction = 0.0

    for phrase in INTENSIFIERS:
        if phrase in text:
            correction += 0.05

    for phrase in NEGATIONS:
        if phrase in text:
            correction -= 0.08

    return round(max(-0.3, min(0.3, correction)), 4)


# ─────────────────────────────────────────────────────────────
# Main PASS scorer
# ─────────────────────────────────────────────────────────────

def compute_pass_score(
    roberta_negative: float,
    lexicon_density: float,
    engagement: float,
    correction: float
) -> float:
    """
    PASS = α·roberta_neg + β·lexicon_density + γ·engagement + δ·correction_clamped

    correction is added directly (can be negative to reduce score).
    Final score clamped to [0, 1].

    Academic citation for this formula structure:
      Weighted ensemble sentiment scoring is established in:
      - Cambria et al. (2013) "New Avenues in Opinion Mining and Sentiment Analysis"
      - Our novel contribution is the engagement amplification component (γ)
        and the pain-specific lexicon (β), neither of which appear in
        general-purpose sentiment literature.
    """
    w = PASS_WEIGHTS
    score = (
        w["roberta"]    * roberta_negative +
        w["lexicon"]    * lexicon_density +
        w["engagement"] * engagement +
        w["correction"] * (correction + 0.3) / 0.6   # normalize [-0.3,+0.3] → [0,1]
    )
    return round(max(0.0, min(1.0, score)), 4)


# ─────────────────────────────────────────────────────────────
# Public API — backward compatible with run_pipeline.py
# ─────────────────────────────────────────────────────────────

def analyze_sentiment(text: str, upvote_score: int = 0) -> dict:
    """
    Main sentiment analysis function.

    Returns all original fields (backward compatible with run_pipeline.py
    and get_nlp_context() in routes.py) PLUS new PASS fields.

    Args:
        text: Post title + content (cleaned or raw — cleaned internally)
        upvote_score: Reddit post score (upvotes). Default 0 = no engagement
                      weighting. Pass post["score"] for full PASS scoring.

    Returns dict with:
        label               : "positive" | "neutral" | "negative"
        compound            : float [-1, 1]  (RoBERTa-derived, VADER-compatible)
        negative            : float [0, 1]   (RoBERTa negative probability)
        neutral             : float [0, 1]   (RoBERTa neutral probability)
        positive            : float [0, 1]   (RoBERTa positive probability)
        complaint_intensity : float [0, 1]   (original field — now PASS score)
        popportunity        : float [0, 1]   (original field — RoBERTa negative)
        is_opportunity      : bool
        pass_score          : float [0, 1]   (NEW — full PASS composite score)
        pass_components     : dict           (NEW — breakdown for explainability)
    """
    cleaned = clean_text(text)

    # ── Component 1: Twitter-RoBERTa ──────────────────────────────────────────
    roberta = _roberta_scores(cleaned)
    neg_prob = roberta["negative"]
    neu_prob = roberta["neutral"]
    pos_prob = roberta["positive"]

    # Compound: VADER-compatible [-1, 1] derived from RoBERTa probs
    # pos - neg gives direction; neutral is treated as muted signal
    compound = round(pos_prob - neg_prob, 3)

    # Sentiment label from RoBERTa (3-class, more nuanced than SST-2)
    if pos_prob > neg_prob and pos_prob > neu_prob:
        label = "positive"
    elif neg_prob > pos_prob and neg_prob > neu_prob:
        label = "negative"
    else:
        label = "neutral"

    # ── Component 2: Pain lexicon density ────────────────────────────────────
    lexicon_density = _pain_lexicon_density(cleaned)

    # ── Component 3: Engagement amplification ────────────────────────────────
    engagement = _engagement_weight(upvote_score)

    # ── Component 4: Intensifier/negation correction ─────────────────────────
    correction = _intensifier_correction(cleaned)

    # ── PASS composite score ──────────────────────────────────────────────────
    pass_score = compute_pass_score(neg_prob, lexicon_density, engagement, correction)

    # ── Backward-compatible fields ────────────────────────────────────────────
    # complaint_intensity now uses PASS score (more accurate than old formula)
    # Old formula was: abs(compound) + keyword_boost — no engagement, no correction
    complaint_intensity = pass_score

    # popportunity: kept as raw RoBERTa negative probability for Equation (1)
    popportunity = neg_prob

    # is_opportunity: threshold on popportunity (Equation 2, unchanged)
    is_opportunity = popportunity >= THRESHOLD

    return {
        # ── Original fields (backward compatible) ──────────────────────────
        "label":               label,
        "compound":            compound,
        "negative":            round(neg_prob, 3),
        "neutral":             round(neu_prob, 3),
        "positive":            round(pos_prob, 3),
        "complaint_intensity": round(complaint_intensity, 3),
        "popportunity":        round(popportunity, 3),
        "is_opportunity":      is_opportunity,

        # ── New PASS fields ────────────────────────────────────────────────
        "pass_score":          pass_score,
        "pass_components": {
            "roberta_negative": round(neg_prob, 3),
            "lexicon_density":  lexicon_density,
            "engagement":       engagement,
            "correction":       correction,
            "weights":          PASS_WEIGHTS,
        }
    }