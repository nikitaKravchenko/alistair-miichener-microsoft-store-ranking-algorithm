import os
import time
import configparser
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from loguru import logger

from model.tools import clean_text_for_tfidf, read_json

config = configparser.ConfigParser()
config.read('config.ini')


def count_words(text: Any) -> int:
    return len(re.findall(r"\w+", text)) if isinstance(text, str) else 0


def _get_output_paths(save_dir: str) -> Tuple[str, str]:
    try:
        features_name = config.get("PREPROCESSING", "R_FEATURES_FILENAME")
        texts_name = config.get("PREPROCESSING", "R_TEXTS_FILENAME")
        features_path = os.path.join(save_dir, features_name)
        texts_path = os.path.join(save_dir, texts_name)
        logger.debug(
            "Config resolved | save_dir='{}' | features='{}' | texts='{}'",
            save_dir, features_path, texts_path
        )
        return features_path, texts_path
    except Exception as e:
        logger.exception("Failed to read output filenames from config.ini: {}", e)
        raise


def extract_features(data: Dict[str, Dict[str, List[Dict]]],
                     save_dir: str,
                     max_queries: Optional[int] = None) -> None:
    start_ts = time.time()
    logger.info("Start review feature extraction | save_dir='{}' | max_queries={}", save_dir, max_queries)

    os.makedirs(save_dir, exist_ok=True)
    output_csv, texts_output_csv = _get_output_paths(save_dir)

    records: List[Dict[str, Any]] = []
    reviews_texts: List[Dict[str, Any]] = []

    total_queries = 0
    total_products = 0
    skipped_empty = 0

    for i, key in enumerate(data):
        if max_queries is not None and i >= max_queries:
            logger.warning("Reached max_queries limit: {}", max_queries)
            break

        total_queries += 1
        products_reviews = data[key]
        if not isinstance(products_reviews, dict):
            logger.warning("Unexpected structure for query='{}' (expected dict). Skipping.", key)
            continue

        for position, (product_id, reviews) in enumerate(products_reviews.items()):
            if not reviews:
                skipped_empty += 1
                continue

            total_products += 1

            ratings = [r.get("rating", 0) for r in reviews]
            positives = [r.get("helpfulPositive", 0) for r in reviews]
            negatives = [r.get("helpfulNegative", 0) for r in reviews]
            titles = [r.get("title", "") for r in reviews]
            texts = [r.get("reviewText", "") for r in reviews]
            dates = pd.to_datetime(
                [r.get("publishedDate") for r in reviews],
                errors="coerce",
                utc=True
            )

            word_counts = [count_words(t) for t in texts]
            title_lengths = [count_words(t) for t in titles]

            records.append({
                "query": key,
                "position": position + 1,
                "productId": product_id,
                "review_count": len(reviews),
                "average_review_rating": np.mean(ratings),
                "median_review_rating": np.median(ratings),
                "positive_votes_sum": sum(positives),
                "negative_votes_sum": sum(negatives),
                "avg_positive_votes": np.mean(positives),
                "avg_negative_votes": np.mean(negatives),
                "percent_empty_reviews": sum(1 for w in word_counts if w == 0) / len(word_counts),
                "percent_reviews_with_title": sum(1 for t in titles if len(t.strip()) > 0) / len(titles),
                "review_count_0_words": sum(1 for w in word_counts if w == 0),
                "review_count_1_50_words": sum(1 for w in word_counts if 1 <= w <= 50),
                "review_count_51_100_words": sum(1 for w in word_counts if 51 <= w <= 100),
                "review_count_101_plus_words": sum(1 for w in word_counts if w > 101),
            })

            for review in reviews:
                reviews_texts.append({
                    "query": key,
                    "position": position + 1,
                    "productId": product_id,
                    "title": clean_text_for_tfidf(review.get('title')),
                    "reviewText": clean_text_for_tfidf(review.get('reviewText'))
                })

    logger.info(
        "Collected review data | queries={} | products={} | skipped_empty={}",
        total_queries, total_products, skipped_empty
    )

    # Features DataFrame
    df_features = pd.DataFrame(records)
    if df_features.empty:
        logger.warning("No review features collected. Output DataFrames are empty; nothing will be saved.")
        return

    # Text DataFrame
    df_texts = pd.DataFrame(reviews_texts)

    logger.info(
        "DataFrames ready | features.shape={} | texts.shape={}",
        tuple(df_features.shape), tuple(df_texts.shape)
    )

    # Save both files
    try:
        df_features.to_csv(output_csv, index=False)
        logger.success("Review features saved -> {}", output_csv)
    except Exception as e:
        logger.exception("Failed to save review features to '{}': {}", output_csv, e)
        raise

    try:
        df_texts.to_csv(texts_output_csv, index=False)
        logger.success("Review text fields saved -> {}", texts_output_csv)
    except Exception as e:
        logger.exception("Failed to save review text fields to '{}': {}", texts_output_csv, e)
        raise

    elapsed = time.time() - start_ts
    logger.info("Review feature extraction finished in {:.2f}s", elapsed)


if __name__ == "__main__":
    logger.info("Loading reviews JSON...")
    data = read_json("../../data/backup/reviews_data.json")
    logger.info("JSON loaded | top-level keys={}", len(data) if hasattr(data, "__len__") else "n/a")
    extract_features(
        data=data,
        save_dir="../../data/preprocessing/"
    )
