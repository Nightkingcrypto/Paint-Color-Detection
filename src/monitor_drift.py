"""Simple script to study model drift using logged predictions.

The FastAPI app logs each prediction into logs/predictions.csv.
This script compares the distribution of predicted top-1 colours
to the original training label distribution.
"""

import csv
import json
from collections import Counter

from loguru import logger

from .config import LABEL_COUNTS_PATH, LOGS_DIR


def load_training_distribution():
    with open(LABEL_COUNTS_PATH, "r", encoding="utf-8") as f:
        counts = json.load(f)
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def load_prediction_distribution():
    pred_file = LOGS_DIR / "predictions.csv"
    counts = Counter()
    total = 0
    if not pred_file.exists():
        logger.warning("No predictions.csv found yet. Use the app to generate logs.")
        return {}
    with open(pred_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            top1 = row["top1_color"]
            counts[top1] += 1
            total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def main():
    train_dist = load_training_distribution()
    pred_dist = load_prediction_distribution()
    if not pred_dist:
        logger.info("No prediction distribution to analyse.")
        return

    logger.info("Training distribution (first 10 colours):")
    for k in list(train_dist.keys())[:10]:
        logger.info(f"{k}: {train_dist[k]:.4f}")

    logger.info("Prediction distribution (first 10 colours):")
    for k in list(pred_dist.keys())[:10]:
        logger.info(f"{k}: {pred_dist.get(k, 0.0):.4f}")

    # Very simple drift indicator: top-5 most common colours in each distribution
    top_train = sorted(train_dist.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_pred = sorted(pred_dist.items(), key=lambda kv: kv[1], reverse=True)[:5]

    logger.info(f"Top training colours: {top_train}")
    logger.info(f"Top predicted colours: {top_pred}")
    logger.info(
        "If the real-world predictions shift heavily away from the training distribution, "
        "you can mention this as model drift in your report."
    )


if __name__ == "__main__":
    main()
