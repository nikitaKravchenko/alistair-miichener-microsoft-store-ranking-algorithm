import os
import time
import configparser
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from model.tools import clean_text_for_tfidf, read_json

config = configparser.ConfigParser()
config.read('config.ini')


def safe_len(field: Any) -> int:
    return len(field) if isinstance(field, list) else 0


def word_count(text: Any) -> int:
    return len(re.findall(r"\w+", text)) if isinstance(text, str) else 0


def _get_output_paths(save_dir: str) -> Tuple[str, str]:
    try:
        features_name = config.get("PREPROCESSING", "P_FEATURES_FILENAME")
        texts_name = config.get("PREPROCESSING", "P_TEXTS_FILENAME")
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


def extract_features(data: Dict[str, Dict[str, Dict]],
                     save_dir: str,
                     max_queries: Optional[int] = None) -> None:
    start_ts = time.time()
    logger.info("Start feature extraction | save_dir='{}' | max_queries={}", save_dir, max_queries)

    os.makedirs(save_dir, exist_ok=True)
    output_csv, texts_output_csv = _get_output_paths(save_dir)

    products: List[Dict[str, Any]] = []
    text_data: List[Dict[str, Any]] = []
    now = pd.Timestamp.now(tz='UTC')

    total_queries = 0
    total_products = 0
    skipped_empty = 0

    # Основний цикл
    for i, key in enumerate(data):
        if max_queries is not None and i >= max_queries:
            logger.warning("Reached max_queries limit: {}", max_queries)
            break

        total_queries += 1
        products_by_id = data[key]
        if not isinstance(products_by_id, dict):
            logger.warning("Unexpected structure for query='{}' (expected dict). Skipping.", key)
            continue

        for position, (product_id, product) in enumerate(products_by_id.items()):
            if not product:
                skipped_empty += 1
                logger.debug("Empty/None product | query='{}' | productId='{}' | position={}",
                             key, product_id, position + 1)
                continue

            total_products += 1
            rating = product.get("productRatings", [{}])[0] if isinstance(product.get("productRatings"), list) else {}

            title = product.get("title", "") or ""
            short_title = product.get("shortTitle", "") or ""
            description = product.get("description", "") or ""
            short_desc = product.get("shortDescription", "") or ""

            products.append({
                "query": key,
                "position": position + 1,
                "productId": product_id,
                "title_length": len(title),
                "shortTitle_length": len(short_title),
                "description_length": len(description),
                "shortDescription_length": len(short_desc),
                "title_word_count": word_count(title),
                "description_word_count": word_count(description),
                "categoryId": product.get("categoryId", ""),
                "subcategoryName": product.get("subcategoryName", ""),
                "averageRating": product.get("averageRating", 0),
                "ratingCount": product.get("ratingCount", 0),
                "price": product.get("price", 0),
                "msrp": (product.get("skusSummary", [{}])[0].get("msrp", 0)
                         if isinstance(product.get("skusSummary"), list) and product.get("skusSummary") else 0),
                "hasAddOns": int(bool(product.get("hasAddOns", False))),
                "hasThirdPartyIAPs": int(bool(product.get("hasThirdPartyIAPs", False))),
                "language": product.get("language", ""),
                "supportedLanguages_count": safe_len(product.get("supportedLanguages", [])),
                "platforms_count": safe_len(product.get("platforms", [])),
                "features_count": safe_len(product.get("features", [])),
                "permissionsRequired_count": safe_len(product.get("permissionsRequired", [])),
                "images_count": safe_len(product.get("images", [])),
                "screenshots_count": safe_len(product.get("screenshots", [])),
                "trailers_count": safe_len(product.get("trailers", [])),
                "approximateSizeInBytes": product.get("approximateSizeInBytes", 0),
                "maxInstallSizeInBytes": product.get("maxInstallSizeInBytes", 0),
                "hasInAppPurchases": int(bool(rating.get("hasInAppPurchases", False))),
                "interactiveElements_count": safe_len(rating.get("interactiveElements", [])) if isinstance(rating, dict) else 0,
                "releaseDateUtc": product.get("releaseDateUtc", ""),
                "lastUpdateDateUtc": product.get("lastUpdateDateUtc", "")
            })

            text_data.append({
                "query": key,
                "productId": product_id,
                "position": position + 1,
                "title": clean_text_for_tfidf(title),
                "shortTitle": clean_text_for_tfidf(short_title),
                "description": clean_text_for_tfidf(description),
                "shortDescription": clean_text_for_tfidf(short_desc)
            })

    logger.info(
        "Collected raw items | queries={} | products={} | skipped_empty={}",
        total_queries, total_products, skipped_empty
    )

    # Features DataFrame
    df = pd.DataFrame(products)
    if df.empty:
        logger.warning("No products collected. Output DataFrames are empty; nothing will be saved.")
        return

    # Дати
    for col in ("releaseDateUtc", "lastUpdateDateUtc"):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

    df["days_since_release"] = (now - df["releaseDateUtc"]).dt.days
    df["days_since_update"] = (now - df["lastUpdateDateUtc"]).dt.days

    # Прості агрегати по датах для діагностики
    try:
        rel_min = df["releaseDateUtc"].min()
        rel_max = df["releaseDateUtc"].max()
        upd_min = df["lastUpdateDateUtc"].min()
        upd_max = df["lastUpdateDateUtc"].max()
        logger.debug(
            "Date spans | release[min={}, max={}] | update[min={}, max={}]",
            rel_min, rel_max, upd_min, upd_max
        )
    except Exception as e:
        logger.warning("Failed to compute date spans: {}", e)

    # Text corpus for further NLP
    text_df = pd.DataFrame(text_data)

    logger.info(
        "DataFrames ready | features.shape={} | texts.shape={}",
        tuple(df.shape), tuple(text_df.shape)
    )

    # Save both files
    try:
        df.to_csv(output_csv, index=False)
        logger.success("Products features saved -> {}", output_csv)
    except Exception as e:
        logger.exception("Failed to save product sfeatures to '{}': {}", output_csv, e)
        raise

    try:
        text_df.to_csv(texts_output_csv, index=False)
        logger.success("Products text fields saved -> {}", texts_output_csv)
    except Exception as e:
        logger.exception("Failed to save products text fields to '{}': {}", texts_output_csv, e)
        raise

    elapsed = time.time() - start_ts
    logger.info("Feature extraction finished in {:.2f}s", elapsed)


if __name__ == "__main__":
    logger.info("Loading source JSON...")
    data = read_json("../../data/backup/products_data.json")
    logger.info("JSON loaded | top-level keys={}", len(data) if hasattr(data, "__len__") else "n/a")
    extract_features(
        data=data,
        save_dir="../../data/preprocessing/"
    )
