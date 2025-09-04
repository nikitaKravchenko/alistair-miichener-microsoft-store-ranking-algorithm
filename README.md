# Microsoft Store Scraper & Modeling Pipeline

This repository implements an **asynchronous pipeline** that:
1) **Collects** products and reviews from the Microsoft Store,  
2) **Stores** raw payloads and curated features in PostgreSQL,  
3) **Preprocesses** and **trains** LightGBM models with Optuna,  
4) **Explains** results with SHAP,
5) **Sends** summary notifications to Telegram.

---


## Project layout

```
.
├─ config.ini
├─ requirements.txt
├─ docker-compose.yml
├─ Dockerfile
├─ main.py
├─ data/
│  ├─ queries.json
└─ model/
   ├─ analize.py
   ├─ bot.py
   ├─ collector.py
   ├─ queries_processor.py
   ├─ tools.py
   ├─ trainer.py
   ├─ trainer_scripts.py
   ├─ preprocessing/
   │  ├─ __init__.py
   │  ├─ load_from_db.py
   │  ├─ products.py
   │  ├─ reviews.py
   │  └─ save_to_db.py
   └─ storage/
      ├─ __init__.py
      ├─ models.py
      └─ persister.py
```

> **Entry point:** `main.py`

---

## Configuration (`config.ini`)

All runtime settings are **committed** to the repository via `config.ini`. Fill real values before building/running.

### `[SETTINGS]`
| Key           | Type | Description                                                   |
|---------------|------|---------------------------------------------------------------|
| `MAX_REVIEWS` | int  | Max number of reviews to fetch per product.                   |
| `MAX_REPEATS` | int  | Crawl retries limit.                                          |
| `PROXY_URL`   | str  | Optional URL to a proxy list; leave empty to disable proxies. |

### `[STORAGE]`
| Key            | Type | Description |
|----------------|------|-------------|
| `DATABASE_URL` | str  | Async SQLAlchemy DSN, e.g. `postgresql+asyncpg://user:pass@host:5432/dbname`. Use host **`db`** if you run via Docker Compose. |

### `[URL]`
Parametric endpoints used by the scraper.

### `[PATH]`
| Key       | Description |
|-----------|-------------|
| `QUERIES` | Path to the search inputs file, e.g. `data/queries.json`. |

### `[PREPROCESSING]`
File names for intermediate artifacts:
- `P_FEATURES_FILENAME`, `P_TEXTS_FILENAME`
- `R_FEATURES_FILENAME`, `R_TEXTS_FILENAME`

These are used by preprocessing and training scripts to read/write CSVs.

### `[NOTIFICATION]`
| Key                 | Description |
|---------------------|-------------|
| `TELEGRAM_TOKEN`    | Telegram bot token. |
| `TELEGRAM_GROUP_ID` | Target chat/group id for notifications. |

> **Security notice:** keep repo private with restricted access.

---

## Input data (`data/queries.json`)

Format:
```jsonc
{
  "queries": [
    "PDF",
    "PDF Editor",
    "CAD",
    "Note taking",
    "Adobe Alternative"
  ]
}
```
- You can also add or remove queries.  
- The system persists queries into the DB table `queries` to ensure idempotent processing.

---

## Database schema (see `model/storage/models.py`)

Tables (PostgreSQL):
- **`queries`** — list of search queries (primary key: `query`).  
- **`product_features`** — curated product‑level features, `data JSONB`.  
- **`product_texts`** — text payloads/embeddings for products, `data JSONB`.  
- **`review_features`** — curated review‑level features, `data JSONB`.  
- **`review_texts`** — text payloads/embeddings for reviews, `data JSONB`.

> JSONB storage keeps raw/structured payloads auditable; add **indexes** on frequently filtered keys or on metadata columns if you extend the schema.

---

## How it works (pipeline stages)

`main.py` orchestrates the full run with **named stages** (visible in logs):

1. **`startup`** — initialize logging and config; generate a short `run_id` for traceability.  
2. **`queries.process`** — load `data/queries.json`, sync to DB (`queries` table), de‑duplicate.  
3. **`queries.collect`** — fetch final query list from DB (ensures consistency across runs).  
4. **`scrape.ms_store`** — async scraping via `aiohttp` with optional proxy rotation:  
   - product search/details,  
   - reviews (paginated up to `MAX_REVIEWS`).  
5. **`io.save_json`** → `io.read_json` — persist raw results to `data/scraper_output/*.json` and reload to validate I/O.  
6. **`preproc.products`** / **`preproc.reviews`** — build features & text representations.  
7. **`db.save_to_db`** — upsert changed features/texts into JSONB tables.  
8. **`db.load_from_db`** — load training datasets back from DB (clean, stable source).  
9. **`train`** — LightGBM training with Optuna HPO; optional TF‑IDF features for text fields.  
10. **`analyze`** — compute SHAP/importance summaries and aggregate reports to `data/analysis/`.  
11. **`notify.telegram`** — send run summary and attachments to the Telegram chat.  

Failures at any stage are logged; if Telegram is configured, a **failure alert** with a log excerpt is sent automatically.

---

## Running with Docker

### Files you need in the repository root
- `Dockerfile`, `docker-compose.yml`, `config.ini`, `requirements.txt`, `requirements-ml.txt`, `constraints.txt`

### Build & run
```bash
docker compose up -d --build
docker compose logs -f app
```
- The Compose file starts **Postgres 16** as `db` and the application as `app`.  
- The app reads **`config.ini`** from the repository.  
- To apply config changes:
```bash
docker compose up -d --force-recreate
```

### Local run (optional)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-ml.txt -c constraints.txt
python -u main.py
```

## Performance & reliability

- **Concurrency** — keep it moderate to avoid 429s; backoff with jitter is applied on network errors.  
- **Proxies** — rotating proxies improve stability; leave `PROXY_URL` empty to disable.  
- **Idempotence** — queries are stored in DB; raw JSON snapshots go to `data/scraper_output/` to make runs reproducible.  
- **Versioning** — prefer separate artifacts per run (`run_id`) if you need historical comparisons.

---

## Modeling details (LightGBM + Optuna)

- **Target & features** — constructed in preprocessing modules under `model/preprocessing/`.  
- **Text** — optional TF‑IDF with dimensionality reduction; see `trainer.py`.  
- **CV strategy** — prefer time‑aware CV or `GroupKFold` by `product_id` to avoid leakage.  
- **HPO** — Optuna with pruners; restrict the search space to impactful params (`num_leaves`, `learning_rate`, `min_data_in_leaf`, `feature_fraction`, `lambda_l1/l2`).  
- **Explainability** — SHAP summaries saved into `data/analysis/` and aggregated by `analize.py`.

---

## Troubleshooting

- **DB connection errors** (`asyncpg.InterfaceError`, `connection refused`)  
  - Ensure the `db` service is **healthy** and `DATABASE_URL` points to `db` (Docker) or your host (local).  
- **HTTP 429 / blocks**  
  - Reduce concurrency, increase backoff, enable proxies, and space requests.  
- **Build issues (LightGBM/SHAP)**  
  - The Docker image installs `libgomp1`. For local builds ensure a compiler toolchain.  
- **Telegram not sending**  
  - Validate token & chat id; ensure the bot has permission to post in the target chat.
