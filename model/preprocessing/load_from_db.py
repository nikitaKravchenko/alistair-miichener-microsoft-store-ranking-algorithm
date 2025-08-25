from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, AsyncSession
from loguru import logger
import pandas as pd
import time

from model.storage import Persister


async def run(engine: AsyncEngine, session: async_sessionmaker[AsyncSession], targets) -> dict[str, pd.DataFrame]:
    db_data: dict[str, pd.DataFrame] = {}
    slog = logger.bind(stage="db_load")
    slog.info(f"Start DB -> DataFrame | targets={len(targets)}")

    for target in targets:
        name = target["name"]
        model = target["model"]
        tlog = logger.bind(stage="db_load", target=name, table=getattr(model, "__tablename__", "?"))
        t0 = time.monotonic()

        try:
            tlog.info("Init Persister & ensure table...")
            db = Persister(engine=engine, session_factory=session, model=model)
            await db.init()

            tlog.info("Fetching all rows...")
            data_scalars = await db.get_all()
            count = len(data_scalars) if data_scalars else 0
            tlog.info(f"Fetched rows: {count}")

            data = [item.data for item in (data_scalars or [])]
            df = pd.DataFrame(data)
            db_data[name] = df

            mem = int(df.memory_usage(deep=True).sum()) if not df.empty else 0
            tlog.success(f"DataFrame built: shape={df.shape} | mem_bytes={mem}")

            if not df.empty:
                cols = list(df.columns)
                tlog.debug(f"Columns({len(cols)}): {cols[:20]}")
                dtypes_sample = {c: str(df[c].dtype) for c in cols[:10]}
                tlog.debug(f"Dtypes sample: {dtypes_sample}")
                tlog.debug(f"Head(2):\n{df.head(2)}")

        except Exception as e:
            tlog.exception(f"Failed to load target: {e!r}")
            db_data[name] = pd.DataFrame()

        finally:
            dt = time.monotonic() - t0
            tlog.info(f"Finished target | elapsed={dt:.2f}s")

    slog.success(f"All targets processed: {len(db_data)}/{len(targets)}")
    return db_data
