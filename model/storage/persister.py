from typing import Any, Iterable, Optional

import tabulate
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, AsyncEngine
from sqlalchemy import select, delete as sa_delete
from sqlalchemy.exc import IntegrityError

from datetime import datetime
from sqlalchemy import select
from sqlalchemy.orm import InstrumentedAttribute
from sqlalchemy.sql.schema import Column
from sqlalchemy.types import Boolean, DateTime, Integer, Float, String

from model.tools import async_timer


def _coerce_value(col: Column, val):
    if val is None:
        return None
    ctype = col.type

    if isinstance(ctype, Boolean):
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(int(val))
        if isinstance(val, str):
            v = val.strip().lower()
            if v in {"1", "true", "t", "yes", "y"}:
                return True
            if v in {"0", "false", "f", "no", "n"}:
                return False
        return bool(val)

    if isinstance(ctype, DateTime):
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except Exception:
                from dateutil import parser
                return parser.isoparse(val)

    # Float/Integer/String — м’які касти
    if isinstance(ctype, Float):
        return float(val)
    if isinstance(ctype, Integer):
        return int(val)
    if isinstance(ctype, String):
        return str(val)

    return val


def _normalize_filters(model, filters: dict) -> dict:
    norm = {}
    for key, val in filters.items():
        col: InstrumentedAttribute = getattr(model, key, None)
        if col is None or not hasattr(col, "property"):
            continue
        column = col.property.columns[0]
        norm[key] = _coerce_value(column, val)
    return norm



class Persister:
    def __init__(self, engine: AsyncEngine, session_factory: async_sessionmaker[AsyncSession], model):
        self.model = model
        self.Session = session_factory
        self.engine = engine
        if self.engine is None:
            raise ValueError("Session factory must be bound to an AsyncEngine")

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(self.model.__table__.create, checkfirst=True)

    async def delete_table(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(self.model.__table__.drop, checkfirst=True)

    # ---- helpers ----
    def _validate_fields(self, filters: dict[str, Any], show_tabulate=False) -> None:
        table_data = []
        valid_columns = {c.name: c.type for c in self.model.__table__.columns}

        for f, value in filters.items():
            if f not in valid_columns:
                raise AttributeError(
                    f"Unknown field '{f}' for {self.model.__name__}. "
                    f"Valid fields are: {', '.join(sorted(valid_columns))}"
                )

            expected_type: TypeEngine = valid_columns[f]
            received_type = type(value).__name__

            table_data.append([
                f,
                expected_type.__class__.__name__,
                received_type
            ])

        if table_data and show_tabulate:
            print(tabulate.tabulate(
                table_data,
                headers=["Field", "Expected Type", "Received Type"],
                tablefmt="grid"
            ))

    def _apply_filters(self, stmt, filters: dict[str, Any]):
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        return stmt

    # ---- CRUD ----
    async def get(self, **filters) -> Optional[Any]:
        self._validate_fields(filters)
        async with self.Session() as session:
            stmt = self._apply_filters(select(self.model), filters).limit(1)
            res = await session.execute(stmt)
            return res.scalar_one_or_none()

    async def get_many(self, limit: Optional[int] = None, **filters) -> list[Any]:
        self._validate_fields(filters)
        async with self.Session() as session:
            stmt = self._apply_filters(select(self.model), filters)
            if limit:
                stmt = stmt.limit(limit)
            res = await session.execute(stmt)
            return list(res.scalars().all())

    async def get_all(self) -> list[Any]:
        async with self.Session() as session:
            res = await session.execute(select(self.model))
            return list(res.scalars().all())

    async def set(self, **data) -> Any:
        async with self.Session() as session:
            try:
                instance = self.model(**data)
                obj = await session.merge(instance)
                await session.flush()
                await session.commit()
                return obj
            except IntegrityError as e:
                await session.rollback()
                raise RuntimeError(f"Integrity error on {self.model.__name__}: {e}") from e

    async def upsert_many(self, data_collection: list[dict]) -> None:
        async with self.Session() as session:
            try:
                for data in data_collection:
                    instance = self.model(data=data)
                    await session.merge(instance)
                await session.flush()
                await session.commit()
            except IntegrityError as e:
                await session.rollback()
                raise RuntimeError(f"Integrity error on batch upsert: {e}") from e

    async def delete(self, **filters) -> int:
        self._validate_fields(filters)
        async with self.Session() as session:
            stmt = sa_delete(self.model)
            for field, value in filters.items():
                stmt = stmt.where(getattr(self.model, field) == value)
            res = await session.execute(stmt)
            await session.commit()
            return res.rowcount or 0

    # @async_timer
    async def get_by_json_keys(self, keys: dict):
        async with self.Session() as session:
            stmt = select(self.model).where(
                self.model.data.cast(JSONB).contains(keys)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

    # @async_timer
    async def exists_by_json_keys(self, keys: dict) -> bool:
        async with self.Session() as session:
            stmt = select(1).where(self.model.data.contains(keys)).limit(1)
            return (await session.execute(stmt)).scalar() is not None
