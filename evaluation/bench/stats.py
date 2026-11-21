"""Sample statistics shared by benchmark and profiler reports."""
import statistics


def sample_stats(values):
    values = list(values)
    if not values:
        raise ValueError("at least one sample is required")
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"samples": values, "median": statistics.median(values), "mean": mean,
            "stdev": stdev, "min": min(values), "max": max(values),
            "cv": stdev / abs(mean) if mean else 0.0}
