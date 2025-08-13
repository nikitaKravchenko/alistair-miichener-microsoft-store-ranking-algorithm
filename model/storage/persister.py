from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

class Persister:
    def __init__(self, db_url: str, model):
        self.db_url = db_url
        self.model = model
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = async_sessionmaker(bind=self.engine, expire_on_commit=False)

    async def init(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(self.model.metadata.create_all)

    async def delete_table(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(self.model.__table__.drop)

    async def get(self, **filters):
        async with self.Session() as session:
            stmt = select(self.model)
            for field, value in filters.items():
                stmt = stmt.where(getattr(self.model, field) == value)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()


    async def set(self, instance):
        async with self.Session() as session:
            async with session.begin():
                try:
                    await session.merge(instance)
                except IntegrityError as e:
                    raise RuntimeError(f"Integrity error: {e}")

    async def get_all(self):
        async with self.Session() as session:
            stmt = select(self.model)
            result = await session.execute(stmt)
            return result.scalars().all()

    async def delete(self, **filters):
        async with self.Session() as session:
            stmt = select(self.model)
            for field, value in filters.items():
                stmt = stmt.where(getattr(self.model, field) == value)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            if instance:
                await session.delete(instance)
                await session.commit()
