import json
import pandas as pd
import numpy as np
import re
from datetime import datetime

from model.tools import clean_text_for_tfidf


def count_words(text):
    return len(re.findall(r"\w+", text)) if isinstance(text, str) else 0

def extract_review_features(json_path: str, output_csv: str = "reviews_features.csv", texts_output_csv: str = "reviews_texts.csv"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    now = pd.Timestamp.now(tz="UTC")
    records = []
    reviews_texts = []
    # data = {"pdf": data["pdf"]}

    max_queries = None

    for i, key in enumerate(data):
        if i == max_queries:
            break
        for position, (product_id, reviews) in enumerate(data[key].items()):
            if not reviews:
                # records.append({
                #     "query": key,
                #     "position": position + 1,
                #     "productId": product_id,
                #     "review_count": 0,
                #     "average_review_rating": 0,
                #     "median_review_rating": 0,
                #     "positive_votes_sum": 0,
                #     "negative_votes_sum": 0,
                #     "avg_positive_votes": 0,
                #     "avg_negative_votes": 0,
                #     # "avg_review_length": 0,
                #     # "avg_title_length": 0,
                #     "percent_empty_reviews": 0,
                #     "percent_reviews_with_title": 0,
                #     "review_count_0_words": 0,
                #     "review_count_1_50_words": 0,
                #     "review_count_51_100_words": 0,
                #     "review_count_101_plus_words": 0,
                # })
                continue

            ratings = [r.get("rating", 0) for r in reviews]
            positives = [r.get("helpfulPositive", 0) for r in reviews]
            negatives = [r.get("helpfulNegative", 0) for r in reviews]
            titles = [r.get("title", "") for r in reviews]
            texts = [r.get("reviewText", "") for r in reviews]
            dates = pd.to_datetime([r.get("publishedDate") for r in reviews], errors="coerce", utc=True)

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
                # "avg_review_length": np.mean(word_counts),
                # "avg_title_length": np.mean(title_lengths),
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

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    df = pd.DataFrame(reviews_texts)
    df.to_csv(texts_output_csv, index=False)
    print(f"✅ Review features saved to {output_csv}")
    print(f"✅ Text fields saved to {texts_output_csv}")


if __name__ == "__main__":
    extract_review_features("../../data/reviews_data.json")
