import asyncio
import configparser
import datetime
import random
import time
from itertools import cycle
from typing import Optional, Dict, Any, List

import aiohttp
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, AsyncSession

from model.queries_processor import list_queries
from model.storage import Persister, Models
from model.tools import compose_aiohttp_proxy

config = configparser.ConfigParser()
config.read("config.ini")

MAX_REVIEWS = config.getint('SETTINGS', 'MAX_REVIEWS')
MAX_RETRIES = config.getint('SETTINGS', 'MAX_RETRIES')
DATE = datetime.datetime.now().date()


# ---------------------------
# Helpers
# ---------------------------

def _sleep_jitter(min_ms: int = 300, max_ms: int = 800) -> float:
    return random.randint(min_ms, max_ms) / 1000.0

def _proxy_str(p: Optional[str]) -> str:
    if not p:
        return "none"
    scheme = p.split("://", 1)[0] if "://" in p else "unknown"
    return f"{scheme}://***"

def _resp_fingerprint(r: aiohttp.ClientResponse) -> str:
    return f"status={r.status} ct={getattr(r, 'content_type', '?')}"


# ---------------------------
# Collect queries
# ---------------------------

async def collect_queries(engine: AsyncEngine, session: async_sessionmaker[AsyncSession]) -> list[str]:
    log = logger.bind(stage="collect_queries")
    log.info("Collecting search queries...")

    query_persister = Persister(engine=engine, session_factory=session, model=Models.Query)
    await query_persister.init()

    queries = await list_queries(query_persister)
    reformed_queries: List[str] = []
    if queries:
        for q in queries:
            if q and q.strip():
                qn = q.strip().replace("  ", " ").replace(" ", "+")
                reformed_queries.append(qn)
    else:
        raise RuntimeError("Add queries first!")

    log.success(f"Collected queries: total={len(reformed_queries)}")
    if reformed_queries[:5]:
        log.debug(f"Sample queries: {reformed_queries[:5]}")
    return reformed_queries


# ---------------------------
# Orchestrator
# ---------------------------

async def collect_microsoft_store(search_queries: list, proxy_cycle: cycle = None):
    log = logger.bind(stage="collect_orchestrator")
    log.info("Collecting data from Microsoft Store...")

    t0 = time.monotonic()
    async with aiohttp.ClientSession() as session:
        queried_search_data = await _collect_search(session, search_queries, proxy_cycle)
        products_data = await _collect_products(session, queried_search_data, proxy_cycle)
        reviews_data = await _collect_reviews(session, queried_search_data, proxy_cycle)
    dt = time.monotonic() - t0

    log.success(f"Data collected. elapsed={dt:.2f}s | queries={len(search_queries)} | "
                f"with_proxies={bool(proxy_cycle)}")
    return queried_search_data, products_data, reviews_data


# ---------------------------
# Search
# ---------------------------

async def _collect_search(session: aiohttp.ClientSession, search_queries: list[str], proxy_cycle: cycle) -> dict:
    search_url = config.get("URL", "SEARCH")
    queried_search_data: Dict[str, List[Dict[str, Any]]] = {}

    stage_log = logger.bind(stage="search")

    stage_log.info("Collecting data from search...")
    t0 = time.monotonic()

    async def _to_next_cursor(query: str, cursor: str = "", depth: int = 0):
        qlog = logger.bind(stage="search", query=query, depth=depth)
        await asyncio.sleep(_sleep_jitter())
        temp_search_url = search_url.format(query=query, cursor=cursor)
        proxy = compose_aiohttp_proxy(next(proxy_cycle)) if proxy_cycle else None
        qlog.debug(f"GET {temp_search_url} | proxy={_proxy_str(proxy)}")

        try:
            async with session.get(temp_search_url, proxy=proxy) as response:
                qlog.debug(f"Response {_resp_fingerprint(response)} for cursor='{cursor}'")
                if response.status == 200 and response.content_type == 'application/json':
                    json_response = await response.json()
                    products = json_response.get("productsList", [])
                    queried_search_data[query].extend(products)

                    total = len(queried_search_data[query])
                    qlog.info(f"Collected products: total={total} (+{len(products)})")

                    nxt = json_response.get("cursor")
                    if nxt:
                        qlog.debug("Next page detected -> recurse")
                        await _to_next_cursor(query, nxt, depth=depth + 1)
                else:
                    text_preview = await response.text()
                    qlog.warning(f"Unexpected response ({response.status}). preview={text_preview[:300]!r}")
                    if depth < 3:
                        qlog.info("Retrying same cursor...")
                        await _to_next_cursor(query, cursor, depth + 1)
                    else:
                        qlog.error("Max retries reached. Marking query as None.")
                        queried_search_data[query] = None
        except asyncio.TimeoutError:
            qlog.error("TimeoutError")
            if depth < 3:
                qlog.info("Retrying after timeout...")
                await _to_next_cursor(query, cursor, depth + 1)
            else:
                queried_search_data[query] = None
        except Exception as e:
            qlog.exception(f"Request failed: {e!r}")
            if depth < 3:
                qlog.info("Retrying after exception...")
                await _to_next_cursor(query, cursor, depth + 1)
            else:
                queried_search_data[query] = None

    for query in search_queries:
        qlog = logger.bind(stage="search", query=query)
        qlog.info(f"Processing query {query}...")
        queried_search_data[query] = []
        await _to_next_cursor(query)

    dt = time.monotonic() - t0
    stage_log.success(f"Search collection finished. elapsed={dt:.2f}s")
    return queried_search_data


# ---------------------------
# Products
# ---------------------------

async def _collect_products(session: aiohttp.ClientSession, queried_search_data: dict, proxy_cycle: cycle) -> dict:
    product_url = config.get("URL", "PRODUCT")

    stage_log = logger.bind(stage="products")
    stage_log.info("Collecting products data...")

    products_data: Dict[str, Dict[str, Any]] = {}
    data_cache: Dict[str, Any] = {}
    t0 = time.monotonic()

    async def _collect_product(query_key: str, product_data: dict, depth: int = 0):
        product_id = product_data.get('productId')
        title = product_data.get('title')
        plog = logger.bind(stage="products", query=query_key, product_id=product_id, depth=depth)

        if not product_id:
            plog.warning("Missing productId, skipping")
            return

        if product_id in data_cache:
            plog.info(f"Cache hit for product ({title})")
            products_data[query_key][product_id] = data_cache[product_id]
            return

        await asyncio.sleep(_sleep_jitter())
        temp_product_url = product_url.format(product_id=product_id)
        proxy = compose_aiohttp_proxy(next(proxy_cycle)) if proxy_cycle else None
        plog.debug(f"GET {temp_product_url} | proxy={_proxy_str(proxy)}")

        try:
            async with session.get(temp_product_url, proxy=proxy) as response:
                plog.debug(f"Response {_resp_fingerprint(response)}")
                if response.status == 200 and response.content_type == 'application/json':
                    json_response = await response.json()
                    if json_response:
                        data_cache[product_id] = json_response
                        products_data[query_key][product_id] = json_response
                        plog.success(f"Collected product: title={title!r}")
                    else:
                        plog.warning("Empty JSON body")
                else:
                    preview = await response.text()
                    plog.warning(f"Unexpected response ({response.status}). preview={preview[:300]!r}")
                    if depth < MAX_RETRIES:
                        plog.info("Retrying...")
                        await _collect_product(query_key, product_data, depth + 1)
                    else:
                        products_data[query_key][product_id] = None
                        plog.error("Max retries reached. Stored None.")
        except asyncio.TimeoutError:
            if depth < MAX_RETRIES:
                plog.warning("TimeoutError. Retrying...")
                await _collect_product(query_key, product_data, depth + 1)
            else:
                products_data[query_key][product_id] = None
                plog.error("Timeout: max retries reached. Stored None.")
        except Exception as e:
            products_data[query_key][product_id] = None
            plog.exception(f"Failed to collect product data: {e!r}")

    for query_key, search_data in (queried_search_data or {}).items():
        qlog = logger.bind(stage="products", query=query_key)
        qlog.info(f"Processing products for query {query_key}...")
        products_data[query_key] = {}

        if not search_data:
            qlog.warning("No search data (None or empty). Skipping query.")
            continue

        for idx, product in enumerate(search_data, 1):
            if not isinstance(product, dict):
                qlog.warning(f"Unexpected product item type: {type(product).__name__}")
                continue
            if idx % 20 == 0:
                qlog.debug(f"Progress: processed={idx}/{len(search_data)}")
            await _collect_product(query_key, product)

    dt = time.monotonic() - t0
    stage_log.success(f"Products collection finished. elapsed={dt:.2f}s | cache_size={len(data_cache)}")
    return products_data


# ---------------------------
# Reviews
# ---------------------------

async def _collect_reviews(session: aiohttp.ClientSession, queried_search_data: dict, proxy_cycle: cycle):
    reviews_url = config.get("URL", "REVIEWS")

    stage_log = logger.bind(stage="reviews")
    stage_log.info("Collecting reviews data...")

    reviews_data: Dict[str, Dict[str, List[dict]]] = {}
    data_cache: Dict[str, List[dict]] = {}
    t0 = time.monotonic()

    async def _to_next_page(query_key: str, product_data: dict, page_number: int = 1, depth: int = 0):
        product_id = product_data.get('productId')
        title = product_data.get('title')
        rlog = logger.bind(stage="reviews", query=query_key, product_id=product_id, page=page_number, depth=depth)

        if not product_id:
            rlog.warning("Missing productId, skip reviews")
            return

        if product_id in data_cache:
            rlog.info(f"Cache hit for reviews: count={len(data_cache[product_id])}")
            reviews_data[query_key][product_id] = list(data_cache[product_id])
            return

        await asyncio.sleep(_sleep_jitter())
        temp_reviews_url = reviews_url.format(product_id=product_id, page_number=page_number, number_of_items=25)
        proxy = compose_aiohttp_proxy(next(proxy_cycle)) if proxy_cycle else None
        rlog.debug(f"GET {temp_reviews_url} | proxy={_proxy_str(proxy)}")

        try:
            async with session.get(temp_reviews_url, proxy=proxy) as response:
                rlog.debug(f"Response {_resp_fingerprint(response)}")
                if response.status == 200 and response.content_type == 'application/json':
                    json_response = await response.json()
                    if json_response:
                        items = json_response.get('items', [])
                        has_more = json_response.get('hasMorePages')
                        next_page = json_response.get('nextPageNumber')
                        total = json_response.get('totalCount')

                        collected_reviews = len(reviews_data[query_key][product_id])

                        reviews_data[query_key][product_id].extend(items)
                        rlog.info(f"Collected reviews: {collected_reviews}/{total} "
                                  f"(+{len(items)}) cap={MAX_REVIEWS} product={title!r}")

                        cup = collected_reviews < MAX_REVIEWS or MAX_REVIEWS == -1
                        if cup and has_more and next_page:
                            await _to_next_page(query_key, product_data, next_page)
                        else:
                            data_cache[product_id] = list(reviews_data[query_key][product_id])[:MAX_REVIEWS]
                    else:
                        rlog.warning("Empty JSON body")
                else:
                    body_preview = await response.text()
                    rlog.warning(f"Unexpected response ({response.status}). preview={body_preview[:300]!r}")
                    if depth < MAX_RETRIES:
                        rlog.info("Retrying...")
                        await _to_next_page(query_key, product_data, page_number, depth + 1)
        except asyncio.TimeoutError:
            if depth < MAX_RETRIES:
                rlog.warning("TimeoutError. Retrying same page...")
                await _to_next_page(query_key, product_data, page_number, depth + 1)
        except Exception as e:
            rlog.exception(f"Failed to collect reviews: {e!r}")

    for query_key, search_data in (queried_search_data or {}).items():
        qlog = logger.bind(stage="reviews", query=query_key)
        qlog.info("Processing reviews for query...")
        reviews_data[query_key] = {}

        if not search_data:
            qlog.warning("No search data (None or empty). Skipping query.")
            continue

        for product_data in search_data:
            pid = product_data.get('productId')
            title = product_data.get('title')
            if not pid:
                qlog.warning("Skip product with missing productId")
                continue

            qlog.info(f"Product reviews start: {title!r} ({pid})")
            reviews_data[query_key][pid] = []
            await _to_next_page(query_key, product_data)

    dt = time.monotonic() - t0
    stage_log.success(f"Reviews collection finished. elapsed={dt:.2f}s | cache_size={len(data_cache)}")
    return reviews_data


# ---------------------------
# Publishers (optional)
# ---------------------------

async def _collect_publishers(session: aiohttp.ClientSession, queried_search_data: dict, proxy_cycle: cycle):
    publisher_url = config.get("URL", "PUBLISHER")

    stage_log = logger.bind(stage="publishers")
    stage_log.info("Collecting publishers data...")

    publishers_data: Dict[str, Dict[str, Any]] = {}
    data_cache: Dict[str, Any] = {}
    t0 = time.monotonic()

    async def _collect_publisher(query_key: str, product_data: dict, depth: int = 0):
        publisher_name = product_data.get('publisherName')
        plog = logger.bind(stage="publishers", query=query_key, publisher=publisher_name, depth=depth)

        if not publisher_name:
            plog.warning("Missing publisherName, skipping")
            return

        if publisher_name in data_cache:
            plog.info("Cache hit for publisher")
            publishers_data[query_key][publisher_name] = data_cache[publisher_name]
            return

        await asyncio.sleep(_sleep_jitter())
        temp_url = publisher_url.format(publisher_name=publisher_name)
        proxy = compose_aiohttp_proxy(next(proxy_cycle)) if proxy_cycle else None
        plog.debug(f"GET {temp_url} | proxy={_proxy_str(proxy)}")

        try:
            async with session.get(temp_url, proxy=proxy) as response:
                plog.debug(f"Response {_resp_fingerprint(response)}")
                if response.status == 200 and response.content_type == 'application/json':
                    json_response = await response.json()
                    if json_response:
                        publishers_data[query_key][publisher_name] = json_response
                        data_cache[publisher_name] = json_response
                        plog.success("Collected publisher data")
                    else:
                        plog.warning("Empty JSON body")
                else:
                    preview = await response.text()
                    plog.warning(f"Unexpected response ({response.status}). preview={preview[:300]!r}")
                    if depth < MAX_RETRIES:
                        plog.info("Retrying...")
                        await _collect_publisher(query_key, product_data, depth + 1)
                    else:
                        publishers_data[query_key][publisher_name] = None
                        plog.error("Max retries reached. Stored None.")
        except asyncio.TimeoutError:
            if depth < MAX_RETRIES:
                plog.warning("TimeoutError. Retrying...")
                await _collect_publisher(query_key, product_data, depth + 1)
            else:
                publishers_data[query_key][publisher_name] = None
                plog.error("Timeout: max retries reached. Stored None.")
        except Exception as e:
            publishers_data[query_key][publisher_name] = None
            plog.exception(f"Failed to collect publisher data: {e!r}")

    for query_key, search_data in (queried_search_data or {}).items():
        qlog = logger.bind(stage="publishers", query=query_key)
        qlog.info("Processing publishers for query...")
        publishers_data[query_key] = {}

        if not search_data:
            qlog.warning("No search data (None or empty). Skipping query.")
            continue

        for product_data in search_data:
            await _collect_publisher(query_key, product_data)

    dt = time.monotonic() - t0
    stage_log.success(f"Publishers collection finished. elapsed={dt:.2f}s | cache_size={len(data_cache)}")
    return publishers_data
