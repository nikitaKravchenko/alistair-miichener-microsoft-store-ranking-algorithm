from loguru import logger
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession, AsyncEngine

from model.storage import Models, Persister
from model.tools import read_json


async def process_queries(filename: str, engine: AsyncEngine, session: async_sessionmaker[AsyncSession]):
    query_persister = Persister(engine=engine, session_factory=session, model=Models.Query)
    await query_persister.init()

    data = read_json(filename)
    if data:
        db_queries = await list_queries(query_persister)
        for query in db_queries:
            if query not in data["queries"]:
                await delete_query(query, query_persister)

        for query in data["queries"]:
            if query not in db_queries:
                await add_query(query, query_persister)
    else:
        raise Exception(f"File doesn't exists or empty: {filename}")

async def list_queries(query_persister) -> list:
    queries = await query_persister.get_all()
    str_queries = []
    if not queries:
        logger.warning("No queries found.")
    else:
        logger.info("Stored Queries:")
        for i, q in enumerate(queries):
            print(f"    {i+1}: {q.query}")
            str_queries.append(q.query)

    return list(str_queries)

async def add_query(query_str: str, query_persister):
    query_str = query_str.strip().lower()

    if await query_persister.get(**{'query': query_str}):
        logger.warning(f"Query already exists: {query_str}")
    else:
        await query_persister.set(query=query_str)
        logger.info(f"Query added: {query_str}")

async def delete_query(query_str, query_persister):
    query_str = query_str.strip().lower()

    if await query_persister.get(**{'query': query_str}):
        await query_persister.delete(**{'query': query_str})
        logger.info(f"Query deleted: {query_str}")
    else:
        logger.warning(f"Query not found: {query_str}")
