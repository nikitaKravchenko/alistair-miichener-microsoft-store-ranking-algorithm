import json
import pandas as pd
import re
from model.tools import clean_text_for_tfidf

def safe_len(field):
    return len(field) if isinstance(field, list) else 0

def word_count(text):
    return len(re.findall(r"\w+", text)) if isinstance(text, str) else 0

def extract_features_from_file(json_path: str, output_csv: str = "products_features.csv", text_csv: str = "products_texts.csv"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    products = []
    text_data = []
    now = pd.Timestamp.now(tz='UTC')
    # data = {"pdf": data["pdf"]}

    max_queries = None

    for i, key in enumerate(data):
        if i == max_queries:
            break
        for position, (product_id, product) in enumerate(data[key].items()):
            if not product:
                print("Product not exists.")
                continue
            rating = product.get("productRatings", [{}])[0]

            title = product.get("title", "")
            short_title = product.get("shortTitle", "")
            description = product.get("description", "")
            short_desc = product.get("shortDescription", "")

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
                "msrp": product.get("skusSummary", [{}])[0].get("msrp", 0),
                "hasAddOns": int(product.get("hasAddOns", False)),
                "hasThirdPartyIAPs": int(product.get("hasThirdPartyIAPs", False)),
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
                "hasInAppPurchases": int(rating.get("hasInAppPurchases", False)),
                "interactiveElements_count": safe_len(rating.get("interactiveElements", [])),
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

    # Features DataFrame
    df = pd.DataFrame(products)
    df["releaseDateUtc"] = pd.to_datetime(df["releaseDateUtc"], errors="coerce", utc=True)
    df["lastUpdateDateUtc"] = pd.to_datetime(df["lastUpdateDateUtc"], errors="coerce", utc=True)
    df["days_since_release"] = (now - df["releaseDateUtc"]).dt.days
    df["days_since_update"] = (now - df["lastUpdateDateUtc"]).dt.days

    # Text corpus for further NLP
    text_df = pd.DataFrame(text_data)

    # Save both files
    df.to_csv(output_csv, index=False)
    text_df.to_csv(text_csv, index=False)
    print(f"✅ Numerical features saved to {output_csv}")
    print(f"✅ Text fields saved to {text_csv}")

if __name__ == "__main__":
    extract_features_from_file("../../data/products_data.json")
