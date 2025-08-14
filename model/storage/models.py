import datetime
from typing import Optional

from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Date, JSON, Integer, UniqueConstraint, DateTime, Float, Boolean, SmallInteger

Base = declarative_base()


class Models:
    class Query(Base):
        __tablename__ = "queries"
        query = Column(String, primary_key=True)

    class ProductFeatures(Base):
        __tablename__ = "product_features"

        id = Column(Integer, primary_key=True, autoincrement=True)
        data = Column(JSONB, nullable=False)


    class ProductTexts(Base):
        __tablename__ = "product_texts"

        id = Column(Integer, primary_key=True, autoincrement=True)
        data = Column(JSONB, nullable=False)

    class ReviewFeatures(Base):
        __tablename__ = "review_features"

        id = Column(Integer, primary_key=True, autoincrement=True)
        data = Column(JSONB, nullable=False)

    class ReviewTexts(Base):
        __tablename__ = "review_texts"

        id = Column(Integer, primary_key=True, autoincrement=True)
        data = Column(JSONB, nullable=False)