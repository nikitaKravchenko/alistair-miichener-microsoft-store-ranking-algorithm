import asyncio
import os
import sys
import time
import uuid
import logging
from contextlib import contextmanager
from itertools import cycle
from urllib.parse import urlparse

from loguru import logger
import configparser

from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, create_async_engine

from model.analize import analyze_and_summarize_outputs
from model.bot import send_notification, send_failure_alert, latest_log_file, read_log_tail
from model.collector import collect_queries, collect_microsoft_store
from model import preprocessing
from model.storage import Models
from model.tools import get_proxies, save_json, read_json
from model.queries_processor import process_queries
from model import trainer_scripts


RUN_ID = "unknown"

config = configparser.ConfigParser()
config.read("config.ini")

PROXY_URL = config.get("SETTINGS", "PROXY_URL", fallback="")
PROXIES = get_proxies(PROXY_URL) if PROXY_URL else []
PROXY_CYCLE = cycle(PROXIES) if PROXIES else None

QUERIES_PATH = config.get("PATH", "QUERIES", fallback="data/queries.csv")
DATABASE_URL = config.get("STORAGE", "DATABASE_URL", fallback="")

TELEGRAM_TOKEN = config.get("NOTIFICATION", "TELEGRAM_TOKEN", fallback="")
TELEGRAM_GROUP_ID = config.get("NOTIFICATION", "TELEGRAM_GROUP_ID", fallback="")

targets = [
    {
        "filename": config.get("PREPROCESSING", "P_FEATURES_FILENAME"),
        "model": Models.ProductFeatures,
        "name": "products_features",
    },
    {
        "filename": config.get("PREPROCESSING", "P_TEXTS_FILENAME"),
        "model": Models.ProductTexts,
        "name": "products_texts",
    },
    {
        "filename": config.get("PREPROCESSING", "R_FEATURES_FILENAME"),
        "model": Models.ReviewFeatures,
        "name": "reviews_features",
    },
    {
        "filename": config.get("PREPROCESSING", "R_TEXTS_FILENAME"),
        "model": Models.ReviewTexts,
        "name": "reviews_texts",
    },
]

csv_files = [
    "data/analysis/model1_global_top.csv",
    "data/analysis/model2_global_top.csv",
    "data/analysis/model3_global_top_terms.csv",
    "data/analysis/model4_global_top_terms.csv",
    "data/analysis/all_models_global_equal_weighted.csv",
]

md_file = "data/analysis/report.md"


def _mask_token(s: str or None, keep: int = 4) -> str:
    if not s:
        return "None"
    s = str(s)
    return ("*" * max(0, len(s) - keep)) + s[-keep:]


def _mask_db_url(u: str or None) -> str:
    if not u:
        return "None"
    try:
        p = urlparse(u)
        host = p.hostname or "?"
        db = (p.path or "").lstrip("/") or "?"
        return f"{p.scheme}://***:***@{host}/{db}"
    except Exception:
        return "masked"


def configure_logging(cfg: configparser.ConfigParser) -> str:
    global RUN_ID
    RUN_ID = uuid.uuid4().hex[:8]

    log_dir = cfg.get("LOGGING", "DIR", fallback="logs")
    os.makedirs(log_dir, exist_ok=True)

    level = cfg.get("LOGGING", "LEVEL", fallback=os.getenv("LOG_LEVEL", "INFO")).upper()
    rotation = cfg.get("LOGGING", "ROTATION", fallback="1 day")
    retention = cfg.get("LOGGING", "RETENTION", fallback="7 days")
    compression = cfg.get("LOGGING", "COMPRESSION", fallback="zip")
    backtrace = cfg.getboolean("LOGGING", "BACKTRACE", fallback=True)
    diagnose = cfg.getboolean("LOGGING", "DIAGNOSE", fallback=True)

    logger.remove()
    logger.configure(extra={"run_id": RUN_ID, "stage": "-"})

    logger.add(
        sys.stdout,
        level=level,
        colorize=True,
        backtrace=backtrace,
        diagnose=diagnose,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
               "run={extra[run_id]} | {extra[stage]: <18} | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
    )
    logger.add(
        os.path.join(log_dir, "app_{time:YYYYMMDD}.log"),
        level=level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | run={extra[run_id]} | {extra[stage]: <18} | "
               "{name}:{function}:{line} - {message}",
    )

    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                lvl = logger.level(record.levelname).name
            except Exception:
                lvl = record.levelno
            logger.bind(stage=getattr(record, "stage", "stdlog")).opt(
                depth=6, exception=record.exc_info
            ).log(lvl, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=level, force=True)
    for noisy in ("asyncio", "aiohttp.client", "urllib3", "sqlalchemy.engine", "sqlalchemy.pool"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    def handle_exception(exc_type, exc, tb):
        logger.opt(exception=(exc_type, exc, tb)).critical("Unhandled exception")

    sys.excepthook = handle_exception
    return RUN_ID


@contextmanager
def stage(name: str):
    t0 = time.monotonic()
    s = logger.bind(stage=name)
    s.info("Start")
    try:
        yield s
        s.success(f"Done | elapsed={time.monotonic()-t0:.2f}s")
    except Exception:
        s.exception(f"Failed | elapsed={time.monotonic()-t0:.2f}s")
        raise


async def main():
    loop = asyncio.get_running_loop()

    def _asyncio_err_handler(_loop, context):
        logger.bind(stage="asyncio").opt(exception=context.get("exception")).error(
            "Loop error: {}", context.get("message")
        )

    loop.set_exception_handler(_asyncio_err_handler)

    current_stage = "startup"

    with stage("engine.init"):
        engine = create_async_engine(
            DATABASE_URL,
            echo=False,
            pool_size=10,
            max_overflow=20,
        )
        SessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
            bind=engine,
            expire_on_commit=False,
        )
        logger.bind(stage="engine.init").info("DB={}", _mask_db_url(DATABASE_URL))

    try:
        current_stage = "queries.process"
        with stage(current_stage):
            await process_queries(filename=QUERIES_PATH, engine=engine, session=SessionLocal)

        current_stage = "queries.collect"
        with stage(current_stage):
            queries = await collect_queries(engine, SessionLocal)
            logger.bind(stage=current_stage).info("queries_count={}", len(queries))

        current_stage = "scrape.ms_store"
        with stage(current_stage):
            queried_search_data, products_data, reviews_data = await collect_microsoft_store(
                queries, PROXY_CYCLE
            )
            q_cnt = len(queried_search_data or {})
            p_cnt = sum(len(v or {}) for v in (products_data or {}).values())
            r_cnt = sum(len(v or []) for d in (reviews_data or {}).values() for v in (d or {}).values())
            logger.bind(stage=current_stage).info(
                "queries={} products={} reviews={}", q_cnt, p_cnt, r_cnt
            )

        current_stage = "io.save_json"
        with stage(current_stage):
            save_json(products_data, "data/scraper_output/products_data.json")
            save_json(reviews_data, "data/scraper_output/reviews_data.json")

        current_stage = "io.read_json"
        with stage(current_stage):
            products_data = read_json("data/scraper_output/products_data.json")
            reviews_data = read_json("data/scraper_output/reviews_data.json")

        current_stage = "preproc.products"
        with stage(current_stage):
            preprocessing.products.extract_features(
                products_data, save_dir="data/preprocessing/"
            )

        current_stage = "preproc.reviews"
        with stage(current_stage):
            preprocessing.reviews.extract_features(
                reviews_data, save_dir="data/preprocessing/"
            )

        current_stage = "db.save_to_db"
        with stage(current_stage):
            await preprocessing.save_to_db.run(
                engine, SessionLocal, targets, save_dir="data/preprocessing/"
            )

        current_stage = "db.load_from_db"
        with stage(current_stage):
            db_data = await preprocessing.load_from_db.run(engine, SessionLocal, targets)
            for t in targets:
                name = t["name"]
                df = db_data.get(name)
                logger.bind(stage=current_stage).info(
                    "df[{}].shape={}", name, getattr(df, "shape", None)
                )

        current_stage = "train"
        with stage(current_stage):
            trainer_scripts.run(db_data)

        current_stage = "analyze"
        with stage(current_stage):
            analyze_and_summarize_outputs(top_n=300)

        current_stage = "notify.telegram"
        with stage(current_stage):
            if TELEGRAM_TOKEN and TELEGRAM_GROUP_ID:
                await send_notification(TELEGRAM_TOKEN, TELEGRAM_GROUP_ID, csv_files, md_file)
            else:
                logger.bind(stage=current_stage).warning(
                    "Skip: missing TELEGRAM_TOKEN or TELEGRAM_GROUP_ID (token={}, chat={})",
                    _mask_token(TELEGRAM_TOKEN),
                    TELEGRAM_GROUP_ID or "None",
                )

    except Exception as e:
        global FAILURE_NOTIFIED
        log_path = latest_log_file(log_dir=config.get("LOGGING", "DIR", fallback="logs"))
        tail = read_log_tail(log_path)
        err_text = f"{type(e).__name__}: {e}"
        if TELEGRAM_TOKEN and TELEGRAM_GROUP_ID and not FAILURE_NOTIFIED:
            try:
                await send_failure_alert(
                    TELEGRAM_TOKEN,
                    TELEGRAM_GROUP_ID,
                    run_id=RUN_ID,
                    stage=current_stage,
                    error_text=err_text,
                    log_tail=tail,
                )
                FAILURE_NOTIFIED = True
            except Exception:
                logger.bind(stage="notify.telegram").exception("Failure alert send error")
        raise
    finally:
        with stage("engine.dispose"):
            await engine.dispose()


if __name__ == "__main__":
    FAILURE_NOTIFIED = False
    run_id = configure_logging(config)
    logger.bind(stage="startup").info(
        "START | run_id={} | proxies={} | queries_path='{}' | db='{}'",
        run_id,
        len(PROXIES),
        QUERIES_PATH,
        _mask_db_url(config.get("STORAGE", "DATABASE_URL", fallback="")),
    )
    try:
        asyncio.run(main())
        logger.bind(stage="shutdown").success("FINISH run_id={}", run_id)
    except Exception:
        if TELEGRAM_TOKEN and TELEGRAM_GROUP_ID and not FAILURE_NOTIFIED:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    send_failure_alert(
                        TELEGRAM_TOKEN,
                        TELEGRAM_GROUP_ID,
                        run_id=run_id,
                        stage="startup",
                        error_text="Startup failure",
                        log_tail=read_log_tail(
                            latest_log_file(log_dir=config.get("LOGGING", "DIR", fallback="logs"))
                        ),
                    )
                )
                FAILURE_NOTIFIED = True
            except Exception:
                logger.bind(stage="notify.telegram").exception("Failure alert send error (startup)")
            finally:
                try:
                    loop.close()
                except Exception:
                    pass
        logger.bind(stage="shutdown").exception("Aborted run_id={}", run_id)
        raise
