def normalize(value, min_v, max_v):
    if max_v == min_v:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def compute_opportunity_scores(
    topic_stats: dict,
    weights: dict = None
):
    """
    topic_stats example:
    {
        topic_id: {
            "demand": int,
            "sentiment": float,
            "trend": float,
            "competition": float
        }
    }
    """

    if weights is None:
        weights = {
            "demand": 0.35,
            "sentiment": 0.25,
            "trend": 0.25,
            "competition": 0.15
        }

    demands = [v["demand"] for v in topic_stats.values()]
    trends = [v["trend"] for v in topic_stats.values()]

    min_d, max_d = min(demands), max(demands)
    min_t, max_t = min(trends), max(trends)

    scores = {}

    for topic_id, stats in topic_stats.items():
        Dk = normalize(stats["demand"], min_d, max_d)
        Ik = abs(stats["sentiment"])        # intensity
        Tk = normalize(stats["trend"], min_t, max_t)
        Ck = 1 - stats["competition"]       # whitespace

        OSk = (
            weights["demand"] * Dk +
            weights["sentiment"] * Ik +
            weights["trend"] * Tk +
            weights["competition"] * Ck
        )

        scores[topic_id] = round(OSk, 4)

    return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
