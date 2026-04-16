from bertopic import BERTopic
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict


def create_topic_model(n_topics: int = 12) -> BERTopic:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    cluster_model = KMeans(
        n_clusters=n_topics,
        random_state=42,
        n_init=10
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model,
        umap_model=None,
        calculate_probabilities=True,
        verbose=False
    )

    return topic_model


def fit_topics(
    topic_model: BERTopic,
    documents: List[str]
) -> Tuple[List[int], Dict[int, list]]:
    if not documents:
        raise ValueError("No documents provided for topic modeling")

    topics, _ = topic_model.fit_transform(documents)

    topic_keywords = {
        topic_id: topic_model.get_topic(topic_id)
        for topic_id in set(topics)
        if topic_id != -1
    }

    return topics, topic_keywords


# 🔥 THIS is what run_pipeline imports
def run_topic_modeling(documents: List[str], n_topics: int = 12):
    topic_model = create_topic_model(n_topics)
    topics, topic_keywords = fit_topics(topic_model, documents)
    return topics, topic_keywords
