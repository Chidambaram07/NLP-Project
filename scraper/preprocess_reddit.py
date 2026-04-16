#Install Requirements
#pip install pymongo spacy
#python -m spacy download en_core_web_sm


from pymongo import MongoClient
import spacy
import re

# -----------------------
# CONFIG
# -----------------------
MONGO_URI = "mongodb+srv://Chidambaramm_db_user:t2pDW9GGu5riRfd@cluster0.knagehn.mongodb.net/reddit_pain_points?retryWrites=true&w=majority&appName=Cluster0"
  # change if using Atlas
DB_NAME = "reddit_pain_points"
COLLECTION_NAME = "posts"

# -----------------------
# LOAD NLP MODEL
# -----------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------
# FILTER RULES
# -----------------------
EXCLUDE_AUTHORS = {"AutoModerator"}
EXCLUDE_KEYWORDS = {
    "megathread", "monthly", "guidelines",
    "who's hiring", "who is hiring",
    "who's looking for work"
}

PAIN_KEYWORDS = {
    "tired", "stuck", "regret", "frustrated", "confused",
    "rejected", "failed", "not working", "pressure",
    "scared", "struggling", "nothing works", "burnout"
}

# -----------------------
# TEXT CLEANING
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\*\*|\*", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# NLP PREPROCESS
# -----------------------
def preprocess_text(text):
    doc = nlp(text)
    return " ".join(
        token.lemma_
        for token in doc
        if token.is_alpha and not token.is_stop
    )

# -----------------------
# FILTER LOGIC
# -----------------------
def is_candidate_post(title, content, author):
    if author in EXCLUDE_AUTHORS:
        return False
    combined = f"{title} {content}".lower()
    return not any(k in combined for k in EXCLUDE_KEYWORDS)

# -----------------------
# PAIN SIGNAL
# -----------------------
def has_pain_signal(text):
    text = text.lower()
    return any(k in text for k in PAIN_KEYWORDS)

# -----------------------
# PIPELINE
# -----------------------
def run_pipeline():
    client = MongoClient(MONGO_URI)
    col = client[DB_NAME][COLLECTION_NAME]

    docs = col.find({"preprocessed": {"$ne": True}})
    count = 0

    for d in docs:
        title = d.get("title", "")
        content = d.get("content", "")
        author = d.get("author", "")

        full_text = f"{title}. {content}"

        update = {
            "full_text": full_text,
            "is_candidate": is_candidate_post(title, content, author),
            "pain_signal": has_pain_signal(full_text),
            "clean_text": clean_text(full_text),
            "processed_text": preprocess_text(clean_text(full_text)),
            "preprocessed": True
        }

        col.update_one({"_id": d["_id"]}, {"$set": update})
        count += 1

    print(f"✅ Preprocessing completed for {count} documents")

if __name__ == "__main__":
    run_pipeline()
