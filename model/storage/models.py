from sqlalchemy import Column, String, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

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