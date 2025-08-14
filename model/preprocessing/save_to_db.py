import configparser
import asyncio
import os
import os.path
import hashlib
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, AsyncEngine

from model.storage import Persister, Models
from model.tools import unique_dicts, dicts_equal, async_timer

config = configparser.ConfigParser()
config.read("config.ini")


def _file_md5(path: str) -> Optional[str]:
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _df_quick_stats(df: pd.DataFrame) -> Dict[str, Any]:
    try:
        nulls = df.isna().sum().to_dict()
        dtypes = {k: str(v) for k, v in df.dtypes.items()}
        return {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "null_sample": {k: nulls[k] for k in list(nulls)[:8]},
            "dtypes_sample": {k: dtypes[k] for k in list(dtypes)[:8]},
        }
    except Exception:
        return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}


@async_timer
async def proceed(df: pd.DataFrame, db: Persister, *, target_name: str, progress_every: int = 1000) -> None:
    log = logger.bind(target=target_name)
    db_content_list: List[Dict[str, Any]] = []

    total = len(df)
    skipped_existing = 0
    errors: List[Tuple[int, str]] = []

    log.info(f"Start proceed(): total_rows={total}")

    for index, row in df.iterrows():
        row_dict = dict(sorted(row.to_dict().items()))
        try:
            db_item = await db.exists_by_json_keys(row_dict)
            if db_item:
                skipped_existing += 1
                if (index + 1) % progress_every == 0:
                    log.debug(f"Progress: {index+1}/{total} | skipped={skipped_existing} | pending_insert={len(db_content_list)}")
                continue
            else:
                db_content_list.append(row_dict)
                log.success(f"Queued new instance ({index+1}/{total}).")
                if (index + 1) % progress_every == 0:
                    log.debug(f"Progress: {index+1}/{total} | skipped={skipped_existing} | pending_insert={len(db_content_list)}")
        except Exception as e:
            errors.append((index, repr(e)))
            log.error(f"Row #{index+1} failed: {e!r}")

    log.info(f"Pre-upsert stats: to_insert={len(db_content_list)} | skipped_existing={skipped_existing} | errors={len(errors)}")

    try:
        await db.upsert_many(data_collection=db_content_list)
        log.success(f"Upsert complete: inserted={len(db_content_list)} | skipped={skipped_existing} | errors={len(errors)}")
    except IntegrityError as ie:
        log.error(f"IntegrityError on upsert_many: {ie!r}. Sample payload: {db_content_list[:1]}")
        raise
    except Exception as e:
        log.exception(f"Unexpected error on upsert_many: {e!r}")
        raise

    if errors:
        preview = errors[:5]
        log.warning(f"Encountered errors for {len(errors)} rows; first 5: {preview}")


@async_timer
async def run(engine: AsyncEngine, session: async_sessionmaker[AsyncSession], save_dir: str) -> None:
    targets = [
        {"filename": config.get("PREPROCESSING", "P_FEATURES_FILENAME"),
         "model": Models.ProductFeatures,
         "name": "product_features"},
        {"filename": config.get("PREPROCESSING", "P_TEXTS_FILENAME"),
         "model": Models.ProductTexts,
         "name": "product_texts"},
        {"filename": config.get("PREPROCESSING", "R_FEATURES_FILENAME"),
         "model": Models.ReviewFeatures,
         "name": "review_features"},
        {"filename": config.get("PREPROCESSING", "R_TEXTS_FILENAME"),
         "model": Models.ReviewTexts,
         "name": "review_texts"},
    ]

    for target in targets:
        target_name = target["name"]
        log = logger.bind(target=target_name)

        df_path = os.path.join(save_dir, target["filename"])
        log.info(f"Target start: path='{df_path}' | exists={os.path.exists(df_path)}")
        if not os.path.exists(df_path):
            log.error("CSV file not found. Skipping target.")
            continue

        try:
            size = os.path.getsize(df_path)
            md5 = _file_md5(df_path)
            log.debug(f"File stats: size_bytes={size} | md5={md5}")
        except Exception as e:
            log.warning(f"Unable to get file stats: {e!r}")

        try:
            df = pd.read_csv(df_path)
            log.info(f"CSV loaded. shape={df.shape}")
            log.debug(f"Head(2):\n{df.head(2)}")
            stats_before = _df_quick_stats(df)
            log.debug(f"DF stats (before replace NaN): {stats_before}")
        except Exception as e:
            log.exception(f"Failed to load CSV: {e!r}")
            continue

        try:
            df = df.replace({np.nan: None})
            stats_after = _df_quick_stats(df)
            log.debug(f"DF stats (after replace NaN): {stats_after}")
        except Exception as e:
            log.exception(f"NA replace failed: {e!r}")
            continue

        db = Persister(engine=engine, session_factory=session, model=target["model"])

        # try:
        #     await db.delete_table()
        #     log.info("Existing table dropped (if existed).")
        # except Exception as e:
        #     log.warning(f"Table drop failed/irrelevant: {e!r}")

        try:
            await db.init()
            log.success("Table ensured/created.")
        except Exception as e:
            log.exception(f"Table init failed: {e!r}")
            continue

        await proceed(df, db, target_name=target_name)

        log.info(f"Target finished: {target_name}")
