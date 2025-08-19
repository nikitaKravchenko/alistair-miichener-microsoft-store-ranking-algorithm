import asyncio
import os
from typing import List, Optional

from loguru import logger
from telegram import Bot, InputMediaDocument
from telegram.request import HTTPXRequest


# --------- helpers ---------

def _chunk_text(text: str, limit: int = 3500) -> List[str]:
    chunks, buf = [], []
    size = 0
    for line in text.splitlines(True):
        if size + len(line) > limit:
            chunks.append("".join(buf))
            buf, size = [line], len(line)
        else:
            buf.append(line)
            size += len(line)
    if buf:
        chunks.append("".join(buf))
    return chunks

def latest_log_file(log_dir: str = "logs", pattern_prefix: str = "app_", suffix: str = ".log") -> Optional[str]:
    if not os.path.isdir(log_dir):
        return None
    files = sorted(
        f for f in os.listdir(log_dir)
        if f.startswith(pattern_prefix) and f.endswith(suffix)
    )
    return os.path.join(log_dir, files[-1]) if files else None

def read_log_tail(path: Optional[str], max_bytes: int = 1500) -> str:
    if not path or not os.path.exists(path):
        return "(no log file found)"
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            tail = f.read().decode(errors="ignore")
        return tail[-max_bytes:]
    except Exception as e:
        logger.bind(stage="bot.logs").warning("Failed to read log tail: {}", e)
        return "(failed to read log tail)"

# --------- public API ---------

async def send_text(token: str, chat_id: str, text: str, parse_mode: str = "Markdown") -> None:
    logger.bind(stage="bot.send_text").info("Preparing to send text | length={}", len(text or ""))
    bot = Bot(token=token)
    chunks = _chunk_text(text, limit=3500)
    logger.bind(stage="bot.send_text").info("Chunks to send: {}", len(chunks))
    for idx, chunk in enumerate(chunks, start=1):
        try:
            await bot.send_message(chat_id=chat_id, text=chunk, parse_mode=parse_mode, disable_web_page_preview=True)
            logger.bind(stage="bot.send_text").debug("Chunk {}/{} sent (len={})", idx, len(chunks), len(chunk))
        except Exception as e:
            logger.bind(stage="bot.send_text").error("Failed to send chunk {}/{}: {}", idx, len(chunks), e)
            raise

async def send_failure_alert(
    token: str,
    chat_id: str,
    *,
    run_id: str,
    stage: str,
    error_text: str,
    log_tail: Optional[str] = None,
) -> None:
    logger.bind(stage="bot.failure_alert").info("Sending failure alert | run_id={} stage={}", run_id, stage)
    tail = log_tail or "(no tail)"
    msg = (
        f"*Pipeline failed* 🔴\n"
        f"run `{run_id}` | stage `{stage}`\n"
        f"```\n{error_text}\n```\n"
        f"```log\n{tail}\n```"
    )
    await send_text(token, chat_id, msg, parse_mode="Markdown")

CAPTION_LIMIT = 1024

async def send_notification(token: str, chat_id: str, csv_paths: list[str], md_path: str):
    logger.bind(stage="bot.notify").info("Preparing notification | files={} md_path={}", len(csv_paths), md_path)
    request = HTTPXRequest(connect_timeout=30.0, read_timeout=300.0, write_timeout=300.0)
    bot = Bot(token=token, request=request)

    description = ""
    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                description = f.read()
            logger.bind(stage="bot.notify").info("Markdown loaded (len={})", len(description))
        except Exception as e:
            logger.bind(stage="bot.notify").warning("Failed to read markdown '{}': {}", md_path, e)
    else:
        logger.bind(stage="bot.notify").warning("Markdown file not found: {}", md_path)

    if len(description) > CAPTION_LIMIT:
        logger.bind(stage="bot.notify").info("Caption trimmed from {} to {}", len(description), CAPTION_LIMIT)
        description = description[:CAPTION_LIMIT]

    media = []
    for i, path in enumerate(csv_paths):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    file_bytes = f.read()
                logger.bind(stage="bot.notify").info("Read file '{}' (bytes={})", os.path.basename(path), len(file_bytes))
                if i == len(csv_paths) - 1 and description:
                    media.append(InputMediaDocument(file_bytes, filename=os.path.basename(path), caption=description, parse_mode="Markdown"))
                else:
                    media.append(InputMediaDocument(file_bytes, filename=os.path.basename(path)))
            except Exception as e:
                logger.bind(stage="bot.notify").error("Failed to read '{}' : {}", path, e)
                raise
        else:
            logger.bind(stage="bot.notify").warning("File not found: {}", path)

    if media:
        logger.bind(stage="bot.notify").info("Sending media group with {} items", len(media))
        try:
            await bot.send_media_group(chat_id=chat_id, media=media)
            logger.bind(stage="bot.notify").success("Media group sent")
        except Exception as e:
            logger.bind(stage="bot.notify").error("send_media_group failed: {}", e)
            raise
    else:
        logger.bind(stage="bot.notify").warning("No media to send")

if __name__ == "__main__":
    TOKEN = "8267583474:AAGXaUdpxO-wLmJSkDjuH16tgLE_DQnymmM"
    CHAT = "-1003024045058"
    csv_files = [
        "../data/analysis/model1_global_top.csv",
        "../data/analysis/model2_global_top.csv",
        "../data/analysis/model3_global_top_terms.csv",
        "../data/analysis/model4_global_top_terms.csv",
        "../data/analysis/all_models_global_equal_weighted.csv",
    ]
    md_file = "../data/analysis/report.md"

    logger.bind(stage="__main__").info("Running manual send twice")
    asyncio.run(send_notification(TOKEN, CHAT, csv_files, md_file))
