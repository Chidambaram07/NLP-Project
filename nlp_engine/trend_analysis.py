import math
from collections import defaultdict
from datetime import datetime
from typing import List, Dict


def analyze_trends(
    topics: List[int],
    timestamps: List[datetime]
) -> Dict[int, float]:
    """
    Computes trend growth G_k as:

    G_k = d/dt log(|o_k(t)|)

    where |o_k(t)| is the weekly mention count
    and derivative is estimated via linear regression slope.
    """

    topic_time_count = defaultdict(lambda: defaultdict(int))

    # Aggregate weekly counts
    for topic, ts in zip(topics, timestamps):
        if topic == -1:
            continue

        if not isinstance(ts, datetime):
            continue

        week_key = ts.strftime("%Y-%W")
        topic_time_count[topic][week_key] += 1

    trend_scores = {}

    for topic, weeks in topic_time_count.items():
        sorted_weeks = sorted(weeks.items())
        counts = [count for _, count in sorted_weeks]

        if len(counts) < 2:
            continue

        # Log transform
        log_counts = [math.log(max(1, c)) for c in counts]

        # Linear regression slope
        n = len(log_counts)
        x = list(range(n))

        mean_x = sum(x) / n
        mean_y = sum(log_counts) / n

        numerator = sum(
            (xi - mean_x) * (yi - mean_y)
            for xi, yi in zip(x, log_counts)
        )
        denominator = sum(
            (xi - mean_x) ** 2
            for xi in x
        )

        if denominator == 0:
            continue

        slope = numerator / denominator

        trend_scores[topic] = round(slope, 3)

    return trend_scores