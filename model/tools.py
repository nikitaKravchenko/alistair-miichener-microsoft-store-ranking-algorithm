import html
import os.path
import re
import time
from pathlib import Path

from random import shuffle
from typing import Any

import orjson
from loguru import logger
from requests import get


def timer(func: callable):
    """ Decorator function to measure the execution time of the input function. """
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.debug("{0} took {1:.4f} seconds".format(func.__name__, end_time - start_time))
        return result
    return wrapper


def async_timer(func: callable):
    """Decorator function to measure the execution time of the async input function."""
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        end_time = time.perf_counter()
        logger.debug("{0} took {1:.4f} seconds".format(func.__name__, end_time - start_time))
        return result
    return wrapper


def get_proxies(url: str) -> list:
    """Get list of proxies from webshare in format host:port:username:password"""
    if url:
        proxies = get(url).text.strip().split("\r\n")
        shuffle(proxies)
        logger.info("\nProxies:\n    {}".format(len(proxies)))
        return proxies
    else:
        return []

def compose_aiohttp_proxy(proxy: str) -> str or None:
    """Converting webshare proxy to aiohttp proxy in format http://username:password@host:port"""
    pattern = r'^([a-zA-Z0-9\.\-]+):(\d+):([^\s:]+):([^\s:]+)$'

    if bool(re.match(pattern, proxy)):
        host, port, username, password = proxy.split(":")
        return f"http://{username}:{password}@{host}:{port}"
    else:
        return None


def read_json(filename: str):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return orjson.loads(f.read())
    else:
        logger.warning(f"JSON {filename} not found!")
        return {}

def save_json(data: dict or list, filename: str):
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    formatted = orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS)
    path.write_bytes(formatted)

    logger.success(f"JSON saved: {filename}")


def clean_text_for_tfidf(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)

    text = text.lower()

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"http\S+|www\.\S+", " ", text)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    text = re.sub(r"[\r\n\t]+", " ", text)

    text = re.sub(r"[^\w\s.,!?\"'’\-]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def unique_dicts(dict_list):
    seen = set()

    def is_new(d):
        key = tuple(sorted(d.items()))
        if key in seen:
            return False
        seen.add(key)
        return True

    return list(filter(is_new, dict_list))



def dicts_equal(d1: dict[str, Any], d2: dict[str, Any]) -> bool:
    """
    Compare two flat dicts for equality (ignores key order).
    NaN is treated as None.
    """
    clean_d1 = {k: (None if _is_nan(v) else v) for k, v in d1.items()}
    clean_d2 = {k: (None if _is_nan(v) else v) for k, v in d2.items()}
    return clean_d1 == clean_d2


def _is_nan(value: Any) -> bool:
    """Check if value is NaN (float only)."""
    try:
        return value != value  # NaN is not equal to itself
    except Exception:
        return False

